function [Z, Out] = dl_sub(Y, X, Dt, Dsq, Lx, gamma, opts)
% Proximal gradient algorithm for solving the subproblem
[n,p] = size(Y); m = size(X,1);

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-5;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e2;     end

Z0 = X; Z = X; lambda = 1/(Lx+m); alpha = lambda*gamma;
lin = Dsq+Lx*eye(m); const = -(Dt*Y+Lx*Z0);

obj0 = 0.5*trace(Z0'*lin*Z0)+trace(Z0'*const)+gamma*sum(sum(abs(Z0)));
obj = obj0;

nstall = 0; start_time = tic;
for k = 1:maxit
	w = (k-1)/(k+2);
	Q = (1+w)*Z-w*Z0; % extrapolation
	GradfQ = lin*Q+const;
	Z0 = Z;
	Z = prox_l1(Q-lambda*GradfQ, alpha);
	
	obj0 = obj;
	obj =  0.5*trace(Z'*lin*Z)+trace(Z'*const)+gamma*sum(sum(abs(Z)));
	relerr = abs(obj-obj0)/(abs(obj0)+1);

	Out.hist_obj(k) = obj;
	Out.hist_rel(k) = relerr;

	crit = relerr < tol;
	if crit; nstall = nstall+1; else nstall = 0; end;
	if nstall>=3; break; end;
	if toc(start_time)>maxT; break; end;
end
Out.iter = k;
end

function X = prox_l1(V, alpha)
	X = max(V-alpha, 0) - max(-V-alpha, 0);
end
