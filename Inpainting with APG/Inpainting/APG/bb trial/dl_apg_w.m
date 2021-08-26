function [D, Out, X] = dl_apg_w(Y, Corr, m, gamma, opts)
% Y = [yi] is the corrupted image, Corr is the mask matrix
% consists of 0/1 entries, 0 for masked pixels.
[n,p] = size(Y); % [m, p] = size(X), 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-4;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e5;     end

if isfield(opts,'D0')        D = opts.D0;         else D = randn(n,m);end
if isfield(opts,'X0')        X = opts.X0;         else X = randn(m,p);end

% normalize D0.
for j=1:m
    D(:,j) = D(:,j)/norm(D(:,j));
end

% computing the projection matrix.
for i=1:p
    M(:,:,i) = diag(Corr(:,i));
end

nrmY = norm(Y, 'fro');
Dt = D'; Dsq = Dt*D;
Xt = X'; Xsq = X*Xt;
Lx = 1; Ld = 1;

obj0 = evaluation(Y, D, X, Corr, gamma);
obj = obj0;

nstall = 0; start_time = tic;
fprintf('Iteration:     ');

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);
    
    % update X
    Lx = norm(Dsq);
    for i=1:p
        GradX(:,i) = D'*(M(:,:,i)*D*X(:,i)-M(:,:,i)*Y(:,i));
    end
    X = X-GradX/Lx;
    X = sign(X).*max(abs(X)-gamma/Lx,0);
    
    Xt = X'; Xsq = X*Xt;
    
    % update D
    Ld = norm(Xsq);
    GradD = zeros(n,m);
    for i = 1:p
        GradD = GradD + M(:,:,i)*(D*X(:,i)-Y(:,i))*X(:,i)';
    end
    D = D-GradD/Ld;
    nrmD = sqrt(sum(D.^2,1));
    id = nrmD>1; s = sum(id);
    if s>0     
        D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
    end
    Dt = D'; Dsq = Dt*D;
    
    obj0 = obj; 
    [obj, res] = evaluation(Y,D,X,Corr,gamma);

    Out.D{k} = D;
    
    % --- diagnostics, reporting, stopping checks ---
    relerr1 = abs(obj-obj0)/(obj0+1);    relerr2 = res/nrmY;
    
    % reporting
    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1;
    Out.hist_rel(2,k) = relerr2;
    
    % check stopping criterion
    crit = relerr1<tol;
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall>=3 || relerr2<tol 
        Out.exit = 1;
        break 
    end
    if toc(start_time)>maxT 
        Out.exit = 2;
        break 
    end;    
end

Lx = norm(Dsq);
for i=1:p
    GradX(:,i) = Dt*(M(:,:,i)*D*X(:,i)-M(:,:,i)*Y(:,i));
end
X = X-GradX/Lx;
X = sign(X).*max(abs(X)-gamma/Lx,0);
fprintf('\n');
Out.iter = k;
end

function [obj, res]  = evaluation(Y, D, X, Corr, gamma)
res = norm(Corr.*(D*X)-Y, 'fro');
obj = 0.5*res^2 + gamma*sum(abs(X(:)));
end 