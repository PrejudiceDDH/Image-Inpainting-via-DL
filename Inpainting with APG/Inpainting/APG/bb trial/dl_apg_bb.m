function [D, Out, X] = dl_apg_bb(Y, Corr, m, gamma, opts)
% Y = [yi] is the corrupted image, Corr is the mask matrix
% consists of 0/1 entries, 0 for masked pixels.
% does not employ extrapolation, only bb
[n,p] = size(Y); % [m, p] = size(X), 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-4;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e5;     end

if isfield(opts,'D0')        D0 = opts.D0;        else D0 = randn(n,m);end
if isfield(opts,'X0')        X0 = opts.X0;        else X0 = randn(m,p);end

% normalize D0.
for j=1:m
    D0(:,j) = D0(:,j)/norm(D0(:,j));
end

% computing the projection matrix.
for i=1:p
    M(:,:,i) = diag(Corr(:,i));
end

nrmY = norm(Y, 'fro');
D = D0; Dm = D0; Dt = D'; Dsq = Dt*D;
X = X0; Xm = X0; Xt = X'; Xsq = X*Xt;
t0 = 1; t = 1; 
% Lx = 1; Ld = 1;
rw = 0.9999;
GradX = zeros(m,p); GradD = zeros(n,m);

obj0 = evaluation(Y, D0, X0, Corr, gamma);
obj = obj0;

Out.redoN = 0; nstall = 0; start_time = tic;
fprintf('Iteration:     ');

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);   
      
    % update X
    if k >= 3
        s_x = X-X_; s_gx = GradX - GradX_;
        step_x = trace(s_x'*s_x)/trace(s_x'*s_gx);
    else
        step_x = 1/norm(D'*D);
    end

    GradX_ = GradX;
    for i=1:p
        GradX(:,i) = D0'*(M(:,:,i)*D0*X0(:,i)-M(:,:,i)*Y(:,i));
    end
    X_= X;
    X = X0-GradX*step_x;
    X = sign(X).*max(abs(X)-gamma*step_x,0);
    
    
    % update D
    if k >= 3
        s_d = D-D_; s_gd = GradD-GradD_;
        step_d = trace(s_d'*s_d)/trace(s_d'*s_gd);
    else
        step_d = 1/norm(X*X');
    end
    
    GradD_ = GradD; GradD = zeros(n,m);
    for i = 1:p
        GradD = GradD + M(:,:,i)*(D0*X0(:,i)-Y(:,i))*X0(:,i)';
    end
    D_ = D;
    D = D0-GradD*step_d;
    nrmD = sqrt(sum(D.^2,1));
    id = nrmD>1; s = sum(id);
    if s>0     
        D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
    end
    
    obj0 = obj; 
    [obj, res] = evaluation(Y,D,X,Corr,gamma);

    Out.D{k} = D;

    X0 = X; D0 = D;
    
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

if k >= 3
    s_x = X-X_; s_gx = GradX - GradX_;
    step_x = s_x'*s_x/(s_x'*s_gx);
else
    step_x = 1/norm(D'*D);
end

GradX_ = GradX;
for i=1:p
    GradX(:,i) = D0'*(M(:,:,i)*D0*X0(:,i)-M(:,:,i)*Y(:,i));
end
X_= X;
X = X0-GradX*step_x;
X = sign(X).*max(abs(X)-gamma*step_x,0);

fprintf('\n');
Out.iter = k;
end

function [obj, res]  = evaluation(Y, D, X, Corr, gamma)
res = norm(Corr.*(D*X)-Y, 'fro');
obj = 0.5*res^2 + gamma*sum(abs(X(:)));
end 