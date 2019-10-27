% figure1e.m
% This tests a pipeline to compare competitor methods.
% This is identical to figure 1f but we only do 2 clusters.

% Pre-requisites:
% Download Sparse Subspace Clustering (ADMM version) from
% http://www.vision.jhu.edu/code/fetchcode.php?id=4
% Put the unpacked directory in the root folder

clearvars;

addpath 'toolbox\'
addpath 'third party tools\'
addpath 'SSC_ADMM_v1.1\' % See pre-requisite

%%

% Simulation Parameters
S       = 2; % number of subspaces (clusters)
d       = 2; % intrinsic dimension
D	    = 2; % embedding dimension
N       = 50; % number of samples
Nvar    = 0.0; % variance in cluster sample size
delta   = 0.0; % noise variation

% Trial Parameters
num_trials = 50;

% Methods evaluated:
methods = {'HiWA','HiWA-SSC','WA','CORAL','SA','ICP'};
% HiWA-SSC  Hierarchical Wasserstein alignment (with Sparse Subspace Clustering)
% HiWA      Hierarchical Wasserstein alignment (with known clusters, unknown labels)
% WA        Wasserstein alignment (no clustering)
% CORAL     Correlation alignment
% SA        Subspace alignment
% ICP       Iterative closest point (modified to allow reflections)

% Parameters for HiWA
HiWAparam.maxiter         = 300;
HiWAparam.tol             = 1e-1;
HiWAparam.mu              = 5e-3;
HiWAparam.shorn.gamma     = 2e-1;
HiWAparam.shorn.maxiter   = 1000;
HiWAparam.display         = 0;
HiWAparam.WAparam.miter     = 100;
HiWAparam.WAparam.sh_gamma  = 1e-1;
HiWAparam.WAparam.sh_miter  = 150;
HiWAparam.WAparam.tol       = 1e-2;
HiWAparam.WAparam.display   = 0;

% Parameters for Wasserstein Alignment Algorithm
WAparam.maxiter       = 200;
WAparam.limit_gamma   = 1e-1;
WAparam.sh_maxiter    = 1000;
WAparam.gamma         = 1e+2;
WAparam.alpha_gamma   = 0.95;
WAparam.display       = 0;

%% Run Trials

% Display options
lims = [-5,5,-5,5];

% Begin trials
rMSE_results = nan(length(methods),num_trials);
for t = 1:num_trials
    % Generate data
    rng(t); [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,D,d,N,Nvar,delta,0);
    rMSE = @(Yhat) norm(Yhat-Rgt*XX,'fro')^2/norm(Rgt*XX,'fro')^2; % relative Mean Square Error

    % Test algorithms
    for m = 1:length(methods)
        % Load previous data
        if t > 1, result = load_result(getFilename(methods{m})); end
        % Run algo
        switch lower(methods{m})
        case 'hiwa'
            % Hierarchical Wasserstein Alignment
            HiWAparam.Rgt = Rgt;
            [result{t}.R] = HiWA(A,X,B,Y,HiWAparam);
            result{t}.Yhat = result{t}.R*XX;
        case 'hiwa-ssc'
            % Hierarchical Wasserstein Alignment with Sparse Subspace Clustering
            HiWAparam.Rgt = Rgt;
            [result{t}.R] = HiWASSC(XX,YY,S,HiWAparam);
            result{t}.Yhat = result{t}.R*XX;
        case 'wa'
            % Wasserstein Alignment
            % Notes: deterministic annealing + soft-correspondence
            [result{t}.R] = WA(XX,YY,Rgt,WAparam);
            result{t}.Yhat = result{t}.R*XX;
        case 'coral'
            % Correlation Alignment
            result{t}.Yhat = CORAL(XX',YY')';
        case 'sa'
            % Subspace Alignment
            % Fernando, et al. (2014). Subspace alignment for domain adaption. ICCV.
            result{t}.R = SA(XX',YY')';
            result{t}.Yhat = result{t}.R*XX;
        case 'icp'
            % Iterative closest point
            result{t}.R = ICP(XX',YY');
            result{t}.Yhat = result{t}.R*XX;
        end
        % Save result
        save_result(result,getFilename(methods{m}));

        % Compute and store rMSE
        rMSE_results(m,t) = rMSE(result{t}.Yhat);
    end
    
    % Display
    figure(1);
    for m = 1:length(methods)
        if m == 1
            subplot(2,1+length(methods),1); cla;
            PlotLabelledData(YY,Ly);
            xlim(lims(1:2)); ylim(lims(3:4));
            title('Ground Truth');
        end

        subplot(2,1+length(methods),1+m); cla;
        result = load_result(getFilename(methods{m})); % load result
        PlotLabelledData(result{t}.Yhat,Ly);
        title([methods{m} ' (rMSE=' num2str(rMSE(result{t}.Yhat)) ')']);
        xlim(lims(1:2)); ylim(lims(3:4));

        subplot(2,1+length(methods),2+length(methods):2*(1+length(methods))); cla;
        plot(rMSE_results','-o','LineWidth',2); legend(methods);
        ylabel('rMSE'); xlabel('Trial');
    end
    drawnow;
end

%% Final Plots

% Aggregate results
rMSE_results = nan(length(methods),num_trials);
for t = 1:num_trials
    rng(t); [A,B,X,Y,XX,YY,Rgt,Lx,Ly] = GenerateSyntheticSubspaceData(S,D,d,N,Nvar,delta,0);
    rMSE = @(Yhat) norm(Yhat-Rgt*XX,'fro')^2/norm(Rgt*XX,'fro')^2;
    for m = 1:length(methods)
        result = load_result(getFilename(methods{m}));
        rMSE_results(m,t) = rMSE(result{t}.Yhat);
    end
end

% Plot options
fontsize = 20;
markersize = 10;
markers = {'o','s','d','p','h','<','>','v','^','x','.','*','+'};
positions = [0.3, 0.3, 0.4, 0.6];

% Plot
idx = (1:num_trials)/num_trials;
spaced_idx = 1:3:num_trials;
fig = figure(2); clf;
set(fig,'DefaultAxesFontSize',fontsize,...
    'Units','Normalized','OuterPosition',positions);
for m = 1:length(methods)
    sorted_rMSE = sort(rMSE_results(m,:));
    h(m,1) = plot(sorted_rMSE,idx,'-','LineWidth',2); hold on;
    h(m,2) = plot(sorted_rMSE(spaced_idx),idx(spaced_idx),markers{m},...
         'MarkerSize',markersize,...
         'MarkerFaceColor',get(h(m),'Color'),'MarkerEdgeColor',get(h(m),'Color'));
end
ylabel('Cumulative probability','Interpreter','LaTeX');
xlabel('Alignment error','Interpreter','LaTeX');
axis tight; ylim([0,1]); xlim([0,2]);
grid on;

%% Functions

function filename = getFilename(method)
filename = ['results\figure1e_' method '.mat'];
end

function result = load_result(filename)
load(filename);
end

function save_result(result,filename)
save(filename);
end