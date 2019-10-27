function R = WA(XX,YY,Rgt,SCAAparam)

% Standard Correspondence Alignment Algorithm
% Applies OT between points XX and YY with decreasing entropy

% Load Parameters
maxiter         = SCAAparam.maxiter;
sh_maxiter      = SCAAparam.sh_maxiter;
gamma           = SCAAparam.gamma;
alpha_gamma     = SCAAparam.alpha_gamma;
limit_gamma     = SCAAparam.limit_gamma;
display         = SCAAparam.display;

% Error metrics
rMSE = @(Rg) norm(Rgt*XX-Rg*XX,'fro')^2/norm(Rgt*XX,'fro')^2; % relative Mean Square Error
RAE = @(Rg) norm(logm(Rgt'*Rg),'fro')/sqrt(2); % Rotation Alignment Error

% Initialize
K = size(XX,1);
R = orth(rand(K));
Nx = size(XX,2);
Ny = size(YY,2);
p = ones(Nx,1)/Nx; % uniform
q = ones(Ny,1)/Ny; % uniform

% Align
for n = 1:maxiter
    % Scheduled annealing
    if gamma > limit_gamma, gamma = gamma * alpha_gamma; end
    % Solve entropy-regularized OT
    Q = Sinkhorn(p,q,R*XX,YY,gamma,sh_maxiter);
    % Solve Rotation
    R = ClosedFormRotationSolver( YY*Q'*XX' );
    % Display
    if display && ~mod(n,display)
        RX = R*XX;
        figure(10);
        subplot(121); cla; PlotLabelledData(R*XX);
        subplot(122); cla; PlotLabelledData(YY);
%         subplot(121); cla; plot3(RX(1,:),RX(2,:),RX(3,:),'o'); hold on; axis square;
%         subplot(122); cla; plot3(YY(1,:),YY(2,:),YY(3,:),'o'); hold on; axis square;
        subplot(121); title(['Iter=' num2str(n) ', \gamma=' num2str(gamma) ', RAE=' num2str(RAE(R)) ', rMSE=' num2str(rMSE(R))]);
        drawnow;
    end
end

end

% Closed form rotation solver
function R = ClosedFormRotationSolver(M)
[U,~,V] = svd(M);
% C = speye(size(U,2)); C(end,end) = det(U*V'); R = U*C*V'; % Rotation
R = U*V'; % Steifel
end

% % Optimal Transport - Sinkhorn
% function [P,d] = Sinkhorn(p,q,X,Y,gamma,maxiter)
% epsilon = 1e-75; % For numerical purposes
% X2 = sum(X.^2,1); Y2 = sum(Y.^2,1);
% C = repmat(Y2,size(X,2),1) + repmat(X2.',1,size(Y,2)) - 2*X.'*Y;
% K = exp(-C/gamma);
% % K = K + epsilon; % For numerical purposes
% b = ones(size(q))/numel(q);
% for k = 1:maxiter
%     a = p ./ (K *b + epsilon); % For numerical purposes
%     b = q ./ (K'*a + epsilon); % For numerical purposes
%     if isnan(a), error('nan found'); end
% end
% P = diag(a)*K*diag(b);
% d = C(:)'*P(:);
% end

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
