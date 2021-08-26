function [D,X,Out] = dl_apg(Y,m,gamma,opts)

[n,p] = size(Y); 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-4;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e3;     end

if isfield(opts,'D0')        D0 = opts.D0;        else D0 = randn(n,m);end
if isfield(opts,'X0')        X0 = opts.X0;        else X0 = randn(m,p);end

% normalize D0
for j=1:m
    D0(:,j) = D0(:,j)/norm(D0(:,j));
end

nrmY = norm(Y, 'fro');
D = D0; Dm = D0; Dt = D'; Dsq = Dt*D;
X = X0; Xm = X0; Xt = X'; Xsq = X*Xt;
t0 = 1; t = 1; Lx = 1; Ld = 1;
rw = 0.9999;

obj0 = 0.5*norm(D0*X0-Y,'fro')^2+gamma*sum(sum(abs(X0)));
obj = obj0;

Out.redoN = 0; nstall = 0; start_time = tic;
fprintf('Iteration:     ');

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);
    
    % update X
    Lx0 = Lx;       Lx = norm(Dsq); 
    GradX = Dt*(D*Xm-Y);
    X = Xm-GradX/Lx;
    X = sign(X).*max(abs(X)-gamma/Lx,0);
    
    Xt = X';
    Xsq = X*Xt;
    
    % update D
    Ld0 = Ld;       Ld = norm(Xsq);      
    GradD = Dm*Xsq-Y*Xt;
    D = Dm-GradD/Ld;
    nrmD = sqrt(sum(D.^2,1));
    id = nrmD>1; s = sum(id);
    if s>0     
        D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
    end

    Dt = D'; Dsq0 = Dsq; Dsq = D*Dt;
    
    res = norm(D*X-Y,'fro');
    obj0 = obj; 
    obj = 0.5*res^2+gamma*sum(abs(X(:)));
    
    if obj>obj0 
        Out.redoN = Out.redoN+1;
        
        % update X
        Lx = norm(Dsq0);
        GradX = D0'*(D0*X0-Y);
        X = X0-GradX/Lx;
        X = sign(X).*max(abs(X)-gamma/Lx,0);
        
        Xt = X';
        Xsq = X*Xt;
        
        % update D
        Ld = norm(Xsq);
        GradD = D0*Xsq-Y*Xt;
        D = D0-GradD/Ld;
        nrmD = sqrt(sum(D.^2,1));
        id = nrmD>1; s = sum(id);
        if s>0     
            D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
        end
        Dt = D'; Dsq = Dt*D;
        
        res = norm(D*X-Y,'fro');
        obj0 = obj; 
        obj = 0.5*res^2+gamma*sum(abs(X(:)));

        Out.D{k} = D;
    end
    % do extrapolation
    t = (1+sqrt(1+4*t0^2))/2;
    w = (t0-1)/t; % extrapolation weight
    wX = min([w,rw*sqrt(Lx0/Lx)]);
    wD = min([w,rw*sqrt(Ld0/Ld)]);
    Xm = X+wX*(X-X0);   Dm = D+wD*(D-D0);
    t0 = t; X0 = X; D0 = D;
    
    % --- diagnostics, reporting, stopping checks ---
    relerr1 = abs(obj-obj0)/(obj0+1);    relerr2 = res/nrmY;
    
    % reporting
    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1;
    Out.hist_rel(2,k) = relerr2;
    
    % check stopping criterion
    crit = relerr1<tol;
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall>=3 || relerr2<tol break; end
    if toc(start_time)>maxT; break; end;    
end
fprintf('\n'); 
Out.iter = k;
