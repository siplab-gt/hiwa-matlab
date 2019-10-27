function [Rg,P,diagnostic] = HiWASSC(XX,YY,K,param)

% Hierarchical Wasserstein Alignment (with Sparse Subspace Clustering)
% XX    Source dataset matrix D x nx
% YY    Target dataset matrix D x ny
% S     Number of clusters

% Parameters
maxiter                 = param.maxiter;
tol                     = param.tol;
mu                      = param.mu;
shorn.gamma             = param.shorn.gamma;
shorn.maxiter           = param.shorn.maxiter;
WAparam                 = param.WAparam;
display                 = param.display;
Rgt                     = param.Rgt;

%% Pre-processing: Clustering via SSC

if K == 1
    Lx = ones(size(XX,2),1);
    Ly = ones(size(YY,2),1);
else
    % Perform Sparse Subspace Clustering
    affine = false; alpha = 3e3; rho = 0.8;
    Lx = SSC_method(XX,affine,alpha,rho,K);
    Ly = SSC_method(YY,affine,alpha,rho,K);
end

%% HiWA Alignment

% Initialization
D = size(XX,1);
Na = K; Nb = K;
% Rg = eye(K); % Orthogonal matrix
Rg = ClosedFormRotationSolver(rand(D));
P = ones(Na,Nb)/(Na*Nb); % uniform
p = ones(Na,1)/Na; % uniform
q = ones(Nb,1)/Nb; % uniform
L = zeros(D,D,Na,Nb); % Lagrangian multipliers
R = zeros(D,D,Na,Nb); % Auxiliary variables
for k = 1:Na*Nb
    L(:,:,k) = zeros(D);
    R(:,:,k) = eye(D);
end
C = zeros(Na,Nb);
Cmax = 0;
[idx_i,idx_j] = meshgrid(1:Na,1:Nb);
idx_i = idx_i(:); idx_j = idx_j(:);

% Scaling
XL = XX/sqrt(D);
YL = YY/sqrt(D);

% Error metrics
rMSE = @(Rg) norm(Rgt*XX-Rg*XX,'fro')^2/norm(Rgt*XX,'fro')^2; % relative Mean Square Error

% Distributed ADMM
for n = 1:maxiter
    % Solve for each Q (in parallel)
    parfor k = 1:Na*Nb
        T = mu/D*(Rg - L(:,:,k));
        [R(:,:,k),~,C(k)] = WAsolver(XL(:,Lx==idx_i(k)),YL(:,Ly==idx_j(k)),P(k),T,WAparam);
    end

    % Solve for P
    P = SinkhornC(p,q,C,shorn.gamma,shorn.maxiter);
    
    % Solve for Global R
    prev_Rg = Rg;
    Rg = ClosedFormRotationSolver(mean(reshape(R+L,[D,D,Na*Nb]),3));
    
    % Update Multipliers
    L = L + R - Rg;
    
    % Save diagnostics
    diagnostic.gamma(n)     = shorn.gamma;
    diagnostic.Rg_norm(n)   = norm(prev_Rg-Rg,'fro');
    diagnostic.rMSE(n)      = rMSE(Rg);

    % Termination Criteria
    terminate = (sum(vec((isnan(P)))) | (diagnostic.Rg_norm(n) <= tol)) & (n>5);
    
    % Display
    if display && (~mod(n,display) | terminate )
    h = figure(100);
    subplot(151); cla; PlotLabelledData(Rg*XX,Lx);
    subplot(151); title(['RX, Iter=' num2str(n) ', \gamma=' num2str(shorn.gamma) ', rMSE=' num2str(diagnostic.rMSE(n)) ]);
    subplot(152); cla; PlotLabelledData(Rgt*XX,Lx); title('RX (Ground truth)');
    subplot(153); semilogy(diagnostic.Rg_norm); xlabel('# iter'); ylabel('Residual Norm');
    subplot(154); imagesc(P,[0,1/max(Na,Nb)]); axis equal tight; title(['Correspondence Matrix, Entropy=' num2str( -sum(P(:).*log(P(:))) )]);
    Cmax = max([vec(C);Cmax]); % scaling to help reduce parameter tuning
    subplot(155); imagesc(C,[0,Cmax]); axis equal tight; title('Distance Matrix'); colorbar;
    drawnow;
    end
    
    if terminate, break; end
end

end

%% Auxiliary Functions

function Lx = SSC_method(XX,affine,alpha,rho,c)
% Run method
C = admmLasso_mat_func(XX,affine,alpha);
CKSym = BuildAdjacency(thrC(C,rho));
Lx = SpectralClustering(CKSym,c);
end

% Simplified WA solver (EMD on low-rank projections)
function [R,Q,distance] = WAsolver(XX,YY,Pij,T,options)
% Parameters
maxiter      = options.miter;
sh_gamma     = options.sh_gamma;
sh_maxiter   = options.sh_miter;
tol          = options.tol;
display      = options.display;
EPS          = eps;
% Initialization
K = size(XX,1);
Nx = size(XX,2);
Ny = size(YY,2);
R = orth(rand(K));
Q = ones(Nx,Ny)/(Nx*Ny);
p = ones(Nx,1)/Nx;
q = ones(Ny,1)/Ny;
sh_gamma_space = logspace(0,-2,maxiter);
% Alternating minimization
for i = 1:maxiter
    % Solve Rotation
    Rprev = R;
    R = ClosedFormRotationSolver( 2*Pij*YY*Q'*XX' + T );
    % Solve entropy-regularized OT
    [Q,distance] = Sinkhorn(p,q,R*XX,YY,sh_gamma/Pij,sh_maxiter);
    % Display
    if display
        emd(i) = distance;
        deltaR(i) = norm(Rprev-R);
        RXX = R*XX;
        h = figure(101);
        subplot(121); cla;
        plot3(0,0,0,'go','LineWidth',3); hold on;
        plot3(RXX(1,:),RXX(2,:),RXX(3,:),'bo');
        plot3(YY(1,:),YY(2,:),YY(3,:),'ro'); axis square;
        title(['Rotated (Iter=' num2str(i) ')']);
        subplot(122); cla;
        plot(emd,'LineWidth',2); hold on; plot(deltaR,'LineWidth',2); 
        xlabel('Iterations'); ylabel('Metrics'); legend({'EMD','deltaR'});
        drawnow; pause(0.2);
    end
    % Termination
    if norm(Rprev-R) <= tol, break; end
end
end

% Closed form rotation solver
function R = ClosedFormRotationSolver(M)
[U,~,V] = svd(M);
% C = speye(size(U,2)); C(end,end) = det(U*V'); R = U*C*V'; % Rotation
R = U*V'; % Steifel
end

% Optimal Transport - Sinkhorn
function [P,d] = Sinkhorn(p,q,X,Y,gamma,maxiter)
X2 = sum(X.^2,1); Y2 = sum(Y.^2,1);
C = repmat(Y2,size(X,2),1) + repmat(X2.',1,size(Y,2)) - 2*X.'*Y;
K = exp(-C/gamma);
b = ones(size(q))/numel(q);
for k = 1:maxiter
    a = p ./ (K *b);
    b = q ./ (K'*a);
    if isnan(a), error('nan found'); end
end
P = diag(a)*K*diag(b);
d = C(:)'*P(:);
end

% Optimal Transport - Sinkhorn
function [P,d] = SinkhornC(p,q,C,gamma,maxiter)
n = length(q);
K = exp(-C/gamma);
b = ones(n,1);
for k = 1:maxiter
    a = p ./ (K *b);
    b = q ./ (K'*a);
end
P = diag(a)*K*diag(b);
d = C(:)'*P(:);
end

function a = vec(A)
a = reshape(A,[],1);
end