function PlotLabelledData(X,L)
if iscell(X)
    XX = X; X = [];
    for l = 1:size(XX,2), X = [X, XX{l}]; end
end
if nargin < 2
    if exist('XX')
        L = [];
        for l = 1:size(XX,2), L = [L, l*ones(1,size(XX{l},2))]; end
    else
        L = ones(1,size(X,2));
    end
end

if size(X,1) == 2
    for l = 1:length(unique(L))
        idx = find(L==l);
        plot(X(1,idx),X(2,idx), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
        hold on; axis square;
    end
else
    Lidx = unique(L);
    for l = 1:length(unique(L))
        idx = find(L==Lidx(l));
        plot3(X(1,idx),X(2,idx),X(3,idx), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
        hold on; axis square;
    end
end
end