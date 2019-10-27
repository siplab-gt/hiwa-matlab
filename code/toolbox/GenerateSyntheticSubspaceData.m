function [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,K,d,N,Nvar,delta,display)

% Generate Synthetic Data lying on union of subspaces

if nargin < 5, display = 0; end

% Randomly generate the space
for s = 1:S
    D(s) = d; % Fix all to dimension d
%     D(s) = ceil(d*rand); % Maximum of dimension d
    A{s} = orth(randn(K,D(s)));
end

% Create a random rotation matrix (ground truth)
Rgt = orth(rand(K));
[V,~,U] = svd(Rgt);
C = speye(size(U,2)); C(end,end) = det(U*V');
Rgt = U*C*V';

% Randomly rotate A to form B
B = A; for i = 1:size(B,2), B{i} = Rgt * B{i}; end 

% Generate distribution parameters
pos = @(x) max(0,x);
for i = 1:size(A,2)
    dist_param{i}.mu = randn(D(i),1);
    ssigma = rand(D(i));
    ssigma = (ssigma+ssigma')/2;
    [Vs,Ds] = eig(ssigma); ssigma = Vs*pos(Ds)*Vs';
    ssigma = ssigma + D(i)*rand*eye(size(ssigma));
    dist_param{i}.sigma = ssigma;
end

% Populate the subspaces
for i = 1:size(A,2)
    c = Nvar*N;
    Nx = ceil(N-c/2 + c*rand);
    Ny = ceil(N-c/2 + c*rand);
%     Nx = ceil(100);
%     Ny = ceil(100);
    % Gaussian distributed
    mu      = dist_param{i}.mu;
    sigma   = dist_param{i}.sigma;
    if D(i) == 1
        X{i} = A{i} * (mu + sigma*randn(Nx,1))';
        Y{i} = B{i} * (mu + sigma*randn(Ny,1))';
    else
        X{i} = A{i} * mvnrnd(mu,sigma,Nx)';
        Y{i} = B{i} * mvnrnd(mu,sigma,Ny)';
    end
    
    % Add noise
    X{i} = X{i} + delta*randn(size(X{i}));
    Y{i} = Y{i} + delta*randn(size(Y{i}));
end

% Create a proxy for rMSE
XX = []; YY = []; Lx = []; Ly = [];
for i = 1:size(A,2)
    XX = [XX, X{i}];
    YY = [YY, Y{i}];
    Lx  = [Lx, i*ones(1,size(X{i},2))];
    Ly  = [Ly, i*ones(1,size(Y{i},2))];
end
rMSE = @(Rg) norm(Rgt*XX-Rg*XX,'fro')^2/norm(Rgt*XX,'fro')^2; 

% Define Rotation Alignment Error
RAE = @(Rg) norm(logm(Rgt'*Rg),'fro')/sqrt(2);

% Display
if display
    figure(1);
    for i = 1:size(A,2)
        subplot(121); plot3(X{i}(1,:),X{i}(2,:),X{i}(3,:),'o'); hold on; axis square;
        subplot(122); plot3(Y{i}(1,:),Y{i}(2,:),Y{i}(3,:),'o'); hold on; axis square;
    end
end

end