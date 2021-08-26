function [D,X,Out] = dl_bcd(Y,m,gamma,opts)
% gamma is a parameter manually set to control sparsity
[n,p] = size(Y); 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-6;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e3;     end

if isfield(opts,'D')         D = opts.D;          else D = randn(n,m);end
if isfield(opts,'X')         X = opts.X;          else X = randn(m,p);end

for j=1:m
    D(:,j) = D(:,j)/norm(D(:,j));
end

nrmY0 = norm(Y, 'fro');
Dt = D'; Dsq = Dt*D;
Xt = X'; Xsq = X*Xt;
Lx = 1; Ld = 0.1;
opts_ = struct([]); % use the default first

obj0 = 0.5*norm(D*X-Y, 'fro')^2+gamma*sum(sum(abs(X)));
obj = obj0;

nstall = 0; start_time = tic;
fprintf('Iteration:     ');

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);
	% update X
    X = dl_sub(Y, X, Dt, Dsq, Lx, gamma, opts_);
    Xt = X'; Xsq = X*Xt;
%%
    % update D
    D = (Y*Xt + Ld*D)*inv(Xsq+Ld*eye(m));
    nrmD = sqrt(sum(D.^2,1));
    ind = nrmD > 1; s = sum(ind);
    if s>0 % Renormalize <--> projection
        D(:,ind) = D(:,ind)*spdiags(nrmD(ind)'.^(-1),0,s,s);
    end
    Dt = D'; Dsq = Dt*D;
%%
    res = norm(D*X-Y, 'fro');
    obj0 = obj;
    obj = 0.5*res^2+gamma*sum(sum(abs(X)));
    relerr1 = abs(obj-obj0)/(obj0+1); relerr2 = res/nrmY0;

    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1;
    Out.hist_rel(2,k) = relerr2;

    crit = relerr1<tol;
    if crit; nstall = nstall+1; else nstall = 0; end;
    if nstall>=3 || relerr2<tol; break; end;
    if toc(start_time)>maxT; break; end;    
    
end
opt.tol = 1e-9; opt.maxit = 5000; opt.maxT = 1e2;
X = dl_sub(Y, X, Dt, Dsq, Lx, gamma, opt);
fprintf('\n'); 
Out.iter = k;
end