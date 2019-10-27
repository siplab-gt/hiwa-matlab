% figure1cd.m
% 
% Alignment performance VS (d and N) while controlling for D
% d = cluster intrinsic dimension
% N = number of datapoints per cluster
% D = embedding dimension
% C = number of clusters                == 5 (fixed)

clearvars;
addpath 'toolbox\'
addpath 'third party tools\'

%% Perform trials

num_trials = 20;

% Define parameters
S       = 5; % number of subspaces (clusters)
d_space = [2,3,4,5]; % latent dimension
K_space = [6,11,17,22]; % embedding dimension
N_space = [12,25,50,100,200]; % number of datapoints
Nvar    = 0.0; % variance in cluster sample size
delta   = 0.0; % noise variation

% Results
rMSE_threshold = 1e-1;
P_threshold = 1e-2;

% Parameters for HiWA
HiWAparam.maxiter           = 300;
HiWAparam.tol               = 1e-1;
HiWAparam.mu                = 5e-3;
HiWAparam.shorn.gamma       = 2e-1;
HiWAparam.shorn.maxiter     = 1000;
HiWAparam.WAparam.miter     = 100;
HiWAparam.WAparam.sh_gamma  = 1e-1;
HiWAparam.WAparam.sh_miter  = 150;
HiWAparam.WAparam.tol       = 1e-2;
HiWAparam.display           = 10;

%% Preliminaries

% Generate legend
for j = 1:length(N_space)
    legend_str{j} = ['$N$=' num2str(N_space(j))];
end

for i = 1:length(d_space)
    exp_str{i} = ['$d$=' num2str(d_space(i)) ', $D$=' num2str(K_space(i)) ', $S$=' num2str(S)];
end

%% Run trials

thr_rMSE_matrix = nan(length(d_space),length(N_space));
thr_P_matrix    = nan(length(d_space),length(N_space));

tic
for i = 1:length(d_space)
% Intialize
results.rMSE    = nan(length(N_space),num_trials);
results.Perror  = nan(length(N_space),num_trials);
for j = 1:length(N_space)
    % Load variables
    d = d_space(i);
    K = K_space(i);
    N = N_space(j);
    
    % run trials
    for t = 1:num_trials
        % Generate data
        rng(t); disp(['d=' num2str(d) ', K=' num2str(K) ', N=' num2str(N) ', Trial #' num2str(t)]);
        [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,K,d,N,Nvar,delta,0);
        HiWAparam.Rgt = Rgt; % Use ground truth for visualization/validation

        % Generate Metrics
        rMSE = @(Rg) norm(Rgt*XX-Rg*XX,'fro')^2/norm(Rgt*XX,'fro')^2; % relative Mean Square Error

        % Run HiWA
        [results.R{j,t},results.P{j,t},results.diagnostic{j,t}] = ...
            HiWA(A,X,B,Y,HiWAparam);

        % Compute and save metrics
        results.rMSE(j,t)   = rMSE(results.R{j,t});
        results.Perror(j,t) = sum(abs(vec(results.P{j,t} - eye(S)/S)));

        % Print results
        disp(['rMSE = ' num2str(results.rMSE(j,t))]);

        % Display time
        disp(['Elasped time = ' num2str(floor(toc/60)) ':' num2str(floor(mod(toc,60)),'%02.f')]);
    end
    % Save Results
    save(['results\figure1cd_d=' num2str(d) ', K=' num2str(K) ', N=' num2str(N) '.mat']);
end
end

%% Generate Figures 1c and 1d

thr_rMSE_matrix = nan(length(d_space),length(N_space));
thr_P_matrix    = nan(length(d_space),length(N_space));
rMSE_median   = nan(length(d_space),length(N_space));
rMSE_error    = nan(length(d_space),length(N_space),2);
Perror_median = nan(length(d_space),length(N_space));
Perror_error  = nan(length(d_space),length(N_space),2);
temp_Perror = nan(length(N_space),num_trials);

% Aggregate result
for i = 1:length(d_space)
% Intialize
results.rMSE = nan(length(N_space),num_trials);
for j = 1:length(N_space)
    % Load variables
    d = d_space(i);
    K = K_space(i);
    N = N_space(j);
    
    % Load results
    filename = ['results\figure1cd_d=' num2str(d) ', K=' num2str(K) ', N=' num2str(N) '.mat'];
    temp_rMSE = get_rMSE(filename);
    temp_Perr = get_P(filename);
    
    % Extract correspondence results
    thr_P_matrix(i,j) = 0;
    for t = 1:num_trials
        temp_Perror(j,t) = sum(abs(vec(temp_Perr{j,t} - eye(S)/S)));
        thr_P_matrix(i,j) = thr_P_matrix(i,j) + double(temp_Perror(j,t) <= P_threshold);
    end
    thr_P_matrix(i,j) = thr_P_matrix(i,j) / num_trials;
    
    % Extract rMSE results
    thr_rMSE_matrix(i,j) = sum( temp_rMSE(j,:) < rMSE_threshold ) / num_trials;
    rMSE_median(i,j) = median(temp_rMSE(j,:));
    rMSE_error(i,j,1) = prctile(temp_rMSE(j,:),75);
    rMSE_error(i,j,2) = prctile(temp_rMSE(j,:),25);
    Perror_median(i,j) = median(temp_Perror(j,:));
    Perror_error(i,j,1) = prctile(temp_Perror(j,:),75);
    Perror_error(i,j,2) = prctile(temp_Perror(j,:),25);
end
end

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
for i = 1:size(line_colors,1), lineprops.col{i} = line_colors(i,:); end

% Plot rMSE at percentiles
fig = figure(1); clf; 
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
mseb(N_space,rMSE_median,rMSE_error,lineprops);
xylims = axis;
xlim([min(N_space), max(N_space)]);
ylim([0,xylims(4)]);
set(gca, 'YScale', 'log', 'XScale', 'log');
yticks(logspace(-2,0,3));
xticks(sort(N_space));
xticklabels(strsplit(num2str(sort(N_space))));
xlabel('Sample size $n$','Interpreter','latex');
ylabel('Alignment error','Interpreter','latex');
grid on;
drawnow;

% Plot Perror at percentiles
fig = figure(2); clf;
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
mseb(N_space,Perror_median,Perror_error,lineprops);
xylims = axis;
xlim([min(N_space), max(N_space)]);
ylim([0,xylims(4)]);
set(gca, 'XScale', 'log');
xticks(sort(N_space));
xticklabels(strsplit(num2str(sort(N_space))));
xlabel('Sample size $n$','Interpreter','latex');
ylabel('Correspondence error','Interpreter','latex');
legend(exp_str,'Location','NorthEast','Interpreter','latex');
grid on;
drawnow;

%% 

function temp_results = get_rMSE(filename)
load(filename); temp_results = results.rMSE;
end

function temp_results = get_P(filename)
load(filename); temp_results = results.P;
end