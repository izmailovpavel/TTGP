clear all, close all, write_fig = 0; N = 30;

% set up the GP
x1 = linspace(-1,1,22)'; x2 = linspace(-1,1,22)';  % construct covariance grid
cov = {{@covSEiso},{@covSEiso}};                % stationary covariance function
mean = {@meanConst}; lik = {@likGauss};     % constant mean, Gaussian likelihood
sf = 1; ell = 0.5; hypcov = log([ell;sf]); hyp.cov = log([ell;sf/2; ell;sf/2]);
hyp.mean = 0.1; sn = 0.1; hyp.lik = log(sn); % mean & likelihood hyperpapameters
xg = {  x1, x2 };                       % plain Kronecker structure
covg = {@apxGrid,cov,xg};                                      % grid covariance
opt.cg_maxit = 500; opt.cg_tol = 1e-5;                          % LCG parameters
inf = @(varargin) infGrid(varargin{:},opt);      % shortcut for inference method

load('data/data.mat');
x = x_tr;
xx = x;
y = y_tr;
fprintf('xx: (%d, %d)\n', size(xx))
fprintf('y: (%d, %d)\n', size(y))
fprintf('xg: (%d, %d)\n', size(xg))

% set up the query
xs = x_te;
fprintf('xs: (%d, %d)\n', size(xs))
par = {mean,covg,lik,x};              % shortcut for Gaussian process parameters
fprintf('Optimise hyperparameters.\n')
hyp = minimize(hyp,@gp,-N,inf,par{:},y);              % optimise hyperparameters
opt.stat = true;                   % show some more information during inference
opt.ndcovs = 25;                    % ask for sampling-based (exact) derivatives


tic, [post,nlZ,dnlZ] = infGrid(hyp,par{:},y,opt); ti = toc; tic  % run inference
[fmu,fs2,ymu,ys2] = post.predict(xs);
fprintf('Inference/prediction took %1.2f/%1.2f[s]\n',ti,toc)
fprintf('fmu: (%d, %d)\n', size(fmu))
l = size(y_te)(1);
mse = norm(y_te - fmu).^2 / l;
fprintf('mse %d\n', mse)
mean_prediction = sum(y_te, 1) / l;
mse_mean = norm(y_te - mean_prediction).^2 / l;
r2 = 1 - mse / mse_mean;
fprintf('r2 %d\n', r2)
