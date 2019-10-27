% figure_parallelism_comparisons.m
% We compare HiWA with and without parallelism.

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
HiWAparam.display           = 0; % Display every # iterations

%% Vary the number of clusters and record runtime

S_space = 3:10;
num_trials = 10;

run_time_wP = nan(num_trials,length(S_space)); % with parallelism
run_time_woP = nan(num_trials,length(S_space)); % without parallelism

for s = 1:length(S_space)
for t = 1:num_trials
    % Generate data
    S = S_space(s);
    rng(t); % for repeatability
    [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,D,d,N,Nvar,delta,0);
    HiWAparam.Rgt = Rgt; % Use ground truth for visualization/validation
    
    % With Parallelism
    tic
    HiWA(A,X,B,Y,HiWAparam);
    run_time_wP(t,s) = toc;
    
    % Without Parallelism
    tic
    HiWA_noparallelism(A,X,B,Y,HiWAparam);
    run_time_woP(t,s) = toc;
    
    % Show running figure
    disp(['S=' num2str(S) ', trial=' num2str(t) ', '...
          'wP=' num2str(run_time_wP(t,s)) ', '...
          'woP=' num2str(run_time_woP(t,s))]);
end
end

save('results\figure_parallelism_comparison.mat')

%% Plot figure

% Plot settings
fontsize = 20;
positions = [0.3, 0.3, 0.4, 0.6];
markersize = 12;
my_markers = {'o','s','d','^','v','*'};
line_colors = ...
    [     0    0.4470    0.7410;
     0.8500    0.3250    0.0980;
     0.4940    0.1840    0.5560;
     0.4660    0.6740    0.1880;
     0.3010    0.7450    0.9330;
     0.6350    0.0780    0.1840 ];

% Plot rMSE at percentiles
fig = figure(1); clf; 
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
lineprops.col{1} = line_colors(1,:);
mseb(S_space,mean(run_time_wP),std(run_time_wP),lineprops);
lineprops.col{1} = line_colors(2,:);
mseb(S_space,mean(run_time_woP),std(run_time_woP),lineprops);
xylims = axis;
xlim([min(S_space), max(S_space)]);
ylim([0,xylims(4)]);
xlabel('Cluster size $S$','Interpreter','latex');
ylabel('Run time (s)','Interpreter','latex');
legend({'With Parallelism','Without Parallelism'},'Location','NorthWest');
grid on;
drawnow;
