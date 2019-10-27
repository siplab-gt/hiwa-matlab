function [Rg,P,diagnostic] = HiWA_noparallelism(A,X,B,Y,param)
% Hierarchical Wasserstein Alignment (HiWA)
% 
% Hierarchical Optimal Transport for Multimodal Distribution Alignment
% Lee, J. and Dabagia, M. and Dyer, E. and Rozell, C.
% http://arxiv.org/abs/1906.11768
% 
% Datasets X and Y containing cluster stucture are aligned using a
% hierarchical optimal transport approach.
% 
% Inputs:
% A                 Cell containing Ka subspaces of dimensions D x d
% X                 Cell containing Ka data clusters of dimensions D x nx
% B                 Cell containing Kb subspaces of dimensions D x d
% Y                 Cell containing Kb clusters of dimensions D x ny
% param             Algorithm parameters
%   .maxiter        Total ADMM iterations
%   .tol            Termination tolerance (stopping criteria)
%   .mu             ADMM (consensus) parameter
%   .shorn          Sinkhorn parameters
%       .gamma      Entropic regularization parameter
%       .maxiter    Number of Sinkhorn iterations
%   .WAparam        Wasserstein Alignment parameters
%       .maxiter    Number of alternating iterations
%       .sh_gamma   Sinkhorn entropic regularization parameter
%       .sh_maxiter Sinkhorn maximum number of iterations
%       .tol        Tolerance
%   .Rgt            Ground truth rotation (if available)
%   .display        Display every # iterations
% 
% Outputs:
% Rg                Orthogonal transformation
% P                 Cluster correspondences
% diagnostic        Struct containing outputs during the solve
%   .Rg_norm        Residual norm (i.e., ||R_t - R_{t-1}||_F^2)
%   .rMSE           Relative Mean Square Error (if Rgt is provided)
% 
% Copyright (c) 2019, John Lee

% Default parameters
maxiter           = 200;
tol               = 1e-2;
mu                = 2e-2;
shorn.gamma       = 2e-1;
shorn.maxiter     = 1000;
WAparam.miter     = 100;
WAparam.sh_gamma  = 1e-1;
WAparam.sh_miter  = 150;
WAparam.tol       = 1e-3;
display           = 10;

% Assign from param if fields exist
if isfield(param,'maxiter'),        maxiter = param.maxiter; end
if isfield(param,'tol'),            tol = param.tol; end
if isfield(param,'mu'),             mu = param.mu; end
if isfield(param.shorn,'gamma'),    shorn.gamma = param.shorn.gamma; end
if isfield(param.shorn,'maxiter'),  shorn.maxiter = param.shorn.maxiter; end
if isfield(param.WAparam,'miter'),  WAparam.miter = param.WAparam.miter; end
if isfield(param.WAparam,'sh_gamma'),   WAparam.sh_gamma = param.WAparam.sh_gamma; end
if isfield(param.WAparam,'sh_miter'),   WAparam.sh_miter = param.WAparam.sh_miter; end
if isfield(param.WAparam,'tol'),    WAparam.tol = param.WAparam.tol; end
if isfield(param,'Rgt'),            Rgt = param.Rgt; end
if isfield(param,'display'),        display = param.display; end

% Initialization
D = size(X{1},1); % embedding (ambient) dimensions
Ka = size(A,2); Kb = size(B,2);
Rg = ClosedFormRotationSolver(rand(D)); % Randomized initialization
P = ones(Ka,Kb)/(Ka*Kb); % uniform
p = ones(Ka,1)/Ka; % uniform
q = ones(Kb,1)/Kb; % uniform
L = zeros(D,D,Ka,Kb); % Lagrangian multipliers
R = zeros(D,D,Ka,Kb); % Auxiliary variables
for k = 1:Ka*Kb
    L(:,:,k) = zeros(D);
    R(:,:,k) = eye(D);
end
C = zeros(Ka,Kb);
Cmax = 0;

% Precompute low-rank projections of points
[idx_i,idx_j] = meshgrid(1:Ka,1:Kb); idx_i = idx_i(:); idx_j = idx_j(:);
XX = []; Lx = []; XL = [];
for i = 1:size(A,2)
    XX = [XX, X{i}]; % original points in high-d space
    Lx = [Lx, i*ones(1,size(X{i},2))];
    XL = [XL, A{i}*A{i}'*X{i}]; % Low-d embedding in High-d space
end
Ly = []; YL = [];
for j = 1:size(B,2)
    Ly = [Ly, j*ones(1,size(Y{j},2))];
    YL = [YL, B{j}*B{j}'*Y{j}]; % Low-d embedding in High-d space
end

% Scaling
XL = XL/sqrt(D);
YL = YL/sqrt(D);

% Error metrics
if exist('Rgt')
    rMSE = @(Rg) norm(Rgt*XX-Rg*XX,'fro')^2/norm(Rgt*XX,'fro')^2; % relative Mean Square Error
end

% Distributed ADMM
for n = 1:maxiter
    % Solve for each Q (in parallel)
    for k = 1:Ka*Kb % without parallelism
    % parfor k = 1:Ka*Kb % with parallelism
        T = mu/D*(Rg - L(:,:,k));
        [R(:,:,k),~,C(k)] = WAsolver(XL(:,Lx==idx_i(k)),YL(:,Ly==idx_j(k)),P(k),T,WAparam);
    end

    % Solve for P
    P = SinkhornC(p,q,C,shorn.gamma,shorn.maxiter);
    
    % Solve for Global R
    prev_Rg = Rg;
    Rg = ClosedFormRotationSolver(mean(reshape(R+L,[D,D,Ka*Kb]),3));
    
    % Update Multipliers
    L = L + R - Rg;
    
    % Save diagnostics
    diagnostic.Rg_norm(n)   = norm(prev_Rg-Rg,'fro');
    if exist('Rgt')
    diagnostic.rMSE(n)      = rMSE(Rg);
    end
    
    % Termination Criteria
    terminate = (sum(vec((isnan(P)))) | (diagnostic.Rg_norm(n) <= tol)) & (n>5);
    
    % Display
    if display && (~mod(n,display) | terminate )
    if exist('Rgt'), NSP = 5; else, NSP = 4; end
    h = figure(100); set(h,'color','w');
    subplot(1,NSP,1); cla; PlotLabelledData(Rg*XX,Lx);
    subplot(1,NSP,1); title(['T(\mu), Iter=' num2str(n)]);
    sp = 2;
    if exist('Rgt')
    subplot(1,NSP,2); cla; PlotLabelledData(Rgt*XX,Lx); title('Ground truth T(\mu)');
    subplot(1,NSP,2); ax = axis; subplot(151); axis(ax);
    sp = 3;
    end
    subplot(1,NSP,sp); cla; semilogy(diagnostic.Rg_norm,'LineWidth',2); xlabel('# iter'); ylabel('Residual Norm');
    if exist('Rgt')
    hold on; semilogy(diagnostic.rMSE,'LineWidth',2); hold off; ylabel(''); legend({'Residual norm','rMSE'},'Location','NorthEast');
    end
    subplot(1,NSP,sp+1); imagesc(P,[0,1/max(Ka,Kb)]); axis equal tight; title(['Cluster Correspondence Matrix']);% ', Entropy=' num2str( -sum(P(:).*log(P(:))) )]);
    Cmax = max([vec(C);Cmax]); % scaling to help reduce parameter tuning
    subplot(1,NSP,sp+2); imagesc(C,[0,Cmax]); axis equal tight; title('Cluster Distance Matrix'); %colorbar;
    drawnow;
    end
    
    if terminate, break; end
end

end

%% Auxiliary Functions

% Wasserstein Alignment
function [R,Q,distance] = WAsolver(XX,YY,Pij,T,options)
% Parameters
maxiter      = options.miter;
sh_gamma     = options.sh_gamma;
sh_maxiter   = options.sh_miter;
tol          = options.tol;
% Initialization
K = size(XX,1);
Nx = size(XX,2);
Ny = size(YY,2);
R = orth(rand(K));
Q = ones(Nx,Ny)/(Nx*Ny);
p = ones(Nx,1)/Nx;
q = ones(Ny,1)/Ny;
% sh_gamma_space = logspace(0,-2,maxiter);
% Alternating minimization
for i = 1:maxiter
%     sh_gamma = sh_gamma_space(i); % Deterministic annealing
    % Solve Rotation
    Rprev = R;
    R = ClosedFormRotationSolver( 2*Pij*YY*Q'*XX' + T );
    % Solve entropy-regularized OT
    [Q,distance] = Sinkhorn(p,q,R*XX,YY,sh_gamma/Pij,sh_maxiter);
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
% epsilon = 1e-100; % BUG!
X2 = sum(X.^2,1); Y2 = sum(Y.^2,1);
C = repmat(Y2,size(X,2),1) + repmat(X2.',1,size(Y,2)) - 2*X.'*Y;
K = exp(-C/gamma);
% K = K + epsilon; % BUG!
b = ones(size(q))/numel(q);
for k = 1:maxiter
%     a = p ./ (K *b + epsilon); % BUG!
%     b = q ./ (K'*a + epsilon); % BUG!
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