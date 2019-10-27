% demo.m
% This program demonstrates Hierarchical Wasserstein Alignment (HiWA) in
% action with synthetic data.
% 
% Copyright (c) 2019, John Lee

clearvars;
addpath 'toolbox\'

% Define parameters
S       = 8; % define the number of subspaces (clusters)
d       = 2; % subspace dimensions
D       = 6; % ambient dimensions
N       = 50; % number of datapoints per cluster
Nvar    = 0.1; % variance in cluster sample size
delta   = 0.01; % noise variation

% HiWA Parameters
HiWAparam.maxiter           = 200; % Maximum ADMM iterations
HiWAparam.tol               = 1e-3; % Termination tolerance
HiWAparam.mu                = 2e-2; % ADMM parameter
HiWAparam.shorn.gamma       = 2e-1; % Entropic regularization parameter
HiWAparam.shorn.maxiter     = 1000; % Maximum sinkhorn iterations
HiWAparam.WAparam.miter     = 100; % Wasserstein Alignment: max iterations
HiWAparam.WAparam.sh_gamma  = 1e-1; % Wasserstein Alignment: entropic reg.
HiWAparam.WAparam.sh_miter  = 150; % Wasserstein Alignment: Sinkhorn iterations
HiWAparam.WAparam.tol       = 1e-2; % Wasserstein Alignment: termination critr.
HiWAparam.display           = 10; % Display every # iterations

% Generate data
rng(1); % for repeatability
[A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,D,d,N,Nvar,delta,0);
HiWAparam.Rgt = Rgt; % Use ground truth for visualization/validation

% Run HiWA
HiWA(A,X,B,Y,HiWAparam);