function R = SA(X,Y)
% Implementation of Subspace Alignment.
%
% Reference: Fernando, et al. (2014). Subspace alignment for domain 
%            adaption. ICCV.
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
% Output:   R        alignment transformation matrix

% Modified from:
% https://github.com/wmkouw/libTLDA/blob/master/matlab/suba.m

% Data shape
[N, D] = size(X);
[M, E] = size(Y);

% Check if dimensionalities are the same
if D~=E; error('Data dimensionalities not the same in both domains'); end

% Principal components of each domain
PX = pca(zscore(X));
PY = pca(zscore(Y));

% Aligned source components
R = PX'*PY;

end