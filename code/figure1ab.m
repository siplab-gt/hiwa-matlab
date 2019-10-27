% figure1ab.m
%
% Alignment performance on equally-spaced subspaces VS randomly-spaced subspaces

clearvars;
addpath 'toolbox\'
addpath 'third party tools\'

%% Perform trials

num_trials = 20;

% Define parameters
S       = 5; % number of subspaces (clusters)
d       = 2; % intrinsic dimension
K	    = 6; % embedding dimension
N_space = [25,50,100]; % number of samples
Nvar    = 0.0; % variance in cluster sample size
delta   = 0.0; % noise variation

% Results
rMSE_threshold = 1e-1;
P_threshold = 1e-2;

% Parameters for HiWA
HiWAparam.maxiter         = 300;
HiWAparam.tol             = 1e-1;
HiWAparam.mu              = 5e-3;
HiWAparam.shorn.gamma     = 2e-1;
HiWAparam.shorn.maxiter   = 1000;
HiWAparam.WAparam.miter     = 100;
HiWAparam.WAparam.sh_gamma  = 1e-1;
HiWAparam.WAparam.sh_miter  = 150;
HiWAparam.WAparam.tol       = 1e-2;
HiWAparam.WAparam.display   = 0;
HiWAparam.display         = 10;

%% Preliminaries

line_colors = ...
    [     0    0.4470    0.7410;
     0.8500    0.3250    0.0980;
     0.4940    0.1840    0.5560;
     0.4660    0.6740    0.1880;
     0.3010    0.7450    0.9330;
     0.6350    0.0780    0.1840 ];
markers = {'o','s','d','*','^','v'};

%% Run trials

tic
% run trials
for i = 1:length(N_space)
% Intialize
N               = N_space(i);
results.rMSE    = nan(2,num_trials);
results.Perror  = nan(2,num_trials);
for t = 1:num_trials
    for j = 1:2
        switch j
        case 1
        % Generate randomly spaced sunspaces
        rng(t); disp(['Randomly spaced, Trial #' num2str(t)]);
        [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,K,d,N,Nvar,delta,0);
        case 2
        % Generate equally spaced subspaces
        rng(t); disp(['Equally spaced, Trial #' num2str(t)]);
        [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateEquallySpacedSyntheticSubspaceData(S,K,d,N,Nvar,delta,0);
        end
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
    save(['results\figure1ab_N=' num2str(N) '.mat']);
end
end

%% Final plot

% Plot settings
fontsize = 20;
markersize = 12;
positions = [0.3, 0.3, 0.4, 0.6];
N_space = [25,100];

% Legend
for i = 1:length(N_space)
    legend_str{i} = ['$n$ = ' num2str(N_space(i))];
end

% Plot rMSE
legend_lines = [];
sidx = 2:3:num_trials;
idx = (1:num_trials)/num_trials;
fig = figure(2); clf;
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
for i = 1:length(N_space)
    filename = ['results\figure1ab_N=' num2str(N_space(i)) '.mat'];
    temp_rMSE_results = get_rMSE(filename);
    for jj = 1:2
        switch jj
            case 1, line_style = '-';
            case 2, line_style = '--';
        end
        srMSE = sort(temp_rMSE_results(jj,:)','ascend');
        h(i,jj) = ...
            semilogx(srMSE',idx/max(idx),...
            line_style,'color',line_colors(i,:),'LineWidth',3); hold on;
        hh(i,jj) = ...
            semilogx(srMSE(sidx)',idx(sidx)/max(idx),...
            markers{i},'color',line_colors(i,:),'LineWidth',3); hold on;
        set(hh(i,jj), 'MarkerSize',markersize, 'MarkerFaceColor', get(hh(i,jj),'Color'));
    end
    legend_lines = [legend_lines hh(i,1)];
end
ylabel('Cumulative probability','Interpreter','LaTeX');
xlabel('Alignment error','Interpreter','LaTeX');
axis tight; ylim([0,1]);
grid on;

% Plot Perror
legend_lines = [];
sidx = 2:2:num_trials;
idx = (1:num_trials)/num_trials;
fig = figure(3); clf;
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
for i = 1:length(N_space)
    filename = ['results\figure1ab_N=' num2str(N_space(i)) '.mat'];
    temp_Perror_results = get_Perror(filename);
    for jj = 1:2
        switch jj
            case 1, line_style = '-';
            case 2, line_style = '--';
        end
        srMSE = sort(temp_Perror_results(jj,:)','ascend');
        h(i,jj) = ...
            plot(srMSE',idx/max(idx),...
            line_style,'color',line_colors(i,:),'LineWidth',3); hold on;
        hh(i,jj) = ...
            plot(srMSE(sidx)',idx(sidx)/max(idx),...
            markers{i},'color',line_colors(i,:),'LineWidth',3); hold on;
        set(hh(i,jj), 'MarkerSize',markersize, 'MarkerFaceColor', get(hh(i,jj),'Color'));
    end
    legend_lines = [legend_lines hh(i,1)];
end
ylabel('Cumulative probability','Interpreter','LaTeX');
xlabel('Correspondence error','Interpreter','LaTeX');
lgd = legend(legend_lines,legend_str,'Location','SouthEast','Interpreter','LaTeX');
title(lgd,{'Random (solid)','Equal (dashed)'},'Interpreter','LaTeX');
axis tight; ylim([0,1]);
grid on;



%%

function temp_results = get_rMSE(filename)
load(filename); temp_results = results.rMSE;
end

function temp_results = get_Perror(filename)
load(filename); temp_results = results.Perror;
end

% This function modifies the subspaces so that they are equally spaced
% We shall assume that K = S + d - 1
function [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateEquallySpacedSyntheticSubspaceData(S,K,d,N,Nvar,delta,display)
[A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,K,d,N,Nvar,delta,display);
Aspace = [A{1},null(A{1}')];
for s = 1:S
    AA = Aspace(:,s:s+d-1);
    BB = Rgt*AA;
    X{s} = AA*A{s}'*X{s};
    Y{s} = Rgt*AA*A{s}'*Rgt'*Y{s};
    XX(:,find(Lx==s)) = X{s};
    YY(:,find(Ly==s)) = Y{s};
    A{s} = AA;
    B{s} = BB;
end
end
