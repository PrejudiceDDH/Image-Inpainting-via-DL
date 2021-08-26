function [D, Out, X] = dl_apg_w_bb(Y, Corr, m, gamma, opts)
% Y = [yi] is the corrupted image, Corr is the mask matrix
% consists of 0/1 entries, 0 for masked pixels.
[n,p] = size(Y); % [m, p] = size(X), 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-5;     end
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
GradX = zeros(m,p); GradD = zeros(n,m);

start_time = tic;
%% The first two steps. To collect D_
for k = 1:3
    alpha_X = -1/norm(D'*D);
    for i=1:p
        GradX(:,i) = D'*(M(:,:,i)*D*X(:,i)-M(:,:,i)*Y(:,i));
    end
    X = X+alpha_X*GradX;
    X = sign(X).*max(abs(X)-gamma*alpha_X,0);

    alpha_D = -1/norm(X*X'); 
    GradD = zeros(n,m);
    for i = 1:p
        GradD = GradD + M(:,:,i)*(D*X(:,i)-Y(:,i))*X(:,i)';
    end
    Dm_ = D; D = D+GradD*alpha_D;

    nrmD = sqrt(sum(D.^2,1));
    id = nrmD>1; s = sum(id);
    if s>0     
        D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
    end
end

alpha_X = 1/norm(D'*D);
for i=1:p
    GradX(:,i) = D'*(M(:,:,i)*D*X(:,i)-M(:,:,i)*Y(:,i));
end
X = X-alpha_X*GradX;
X = sign(X).*max(abs(X)-gamma*alpha_X,0);
%%

obj0 = evaluation(Y, D, X, Corr, gamma);
obj = obj0;

nstall = 0; 
fprintf('Iteration:     ');

Q = 1; rho = 1.1; C = obj; delta = 0.8; eta = 0.1;
X0 = X; Xm = X; 
D0 = D; Dm = D; 
alpha_X = 1; alpha_D = 1; t0 = 1; rw = 0.9999;

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);
%%  update D, s_gd needs the gradient information at the updated X
%   perform the BB-method on this step + line search       
    
    GradD_ = zeros(n,m);
    for i = 1:p
        GradD_ = GradD_ + M(:,:,i)*(Dm_*X(:,i)-Y(:,i))*X(:,i)';
    end
    GradD = zeros(n,m);
    for i = 1:p
        GradD = GradD + M(:,:,i)*(Dm*X(:,i)-Y(:,i))*X(:,i)';
    end
    s_d = Dm-Dm_; s_gd = GradD-GradD_;
    alpha_D0 = alpha_D; alpha_D = trace(s_d'*s_d)/trace(s_d'*s_gd);
    
    % non-monotone line search
    stopping = 0;
    GradDsq = trace(GradD'*GradD);
    value = evaluation(Y, Dm+alpha_D*GradD, X, Corr, gamma);
    state0 = (value <= C + delta*alpha_D*GradDsq);
    % state0 == 1, need increase; state0 == 0, need decrease
    while stopping == 0
        if state0 == 1
            alpha_D = alpha_D*rho;
        else
            alpha_D = alpha_D/rho;
        end
        value_ = value;
        value = evaluation(Y, Dm+alpha_D*GradD, X, Corr, gamma);
        state = (value <= C + delta*alpha_D*GradDsq);
        if state == state0
            continue
        else
            stopping = 1;
        end
    end
    if state0 == 1
        alpha_D = alpha_D/rho;
        value = value_;
    end

    D = Dm+alpha_D*GradD; 
    nrmD = sqrt(sum(D.^2,1));
    id = nrmD>1; s = sum(id);
    if s>0     
        D(:,id) = D(:,id)*spdiags(nrmD(id)'.^(-1),0,s,s);
    end
    % update Q for the next iteration.
    Q_ = Q; Q = eta*Q+1;
    C = (eta*Q_*C+value)/Q;

%%
    % Upate X.
    for i=1:p
        GradX(:,i) = D'*(M(:,:,i)*D*Xm(:,i)-M(:,:,i)*Y(:,i));
    end
    alpha_X0 = alpha_X; alpha_X = -1/norm(D'*D);
    X = Xm+alpha_X*GradX;
    X = sign(X).*max(abs(X)-gamma*alpha_X,0);


%%
    t = (1+sqrt(1+4*t0^2))/2;
%     w = (t0-1)/t; % extrapolation weight
    w = 0;
    wX = min([w,rw*sqrt(alpha_X0/alpha_X)]);
    wD = min([w,rw*sqrt(alpha_D0/alpha_D)]);
    Dm_ = Dm;
    Xm = X+wX*(X-X0);   Dm = D+wD*(D-D0);
    t0 = t; X0 = X; D0 = D;

    % --- diagnostics, reporting, stopping checks ---
    obj0 = obj; 
    [obj, res] = evaluation(Y,D,X,Corr,gamma);
    Out.D{k} = D;
    relerr1 = abs(obj-obj0)/(obj0+1); % relative decrease of objective value    
    relerr2 = res/nrmY; % remainder
    
    % reporting
    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1;
    Out.hist_rel(2,k) = relerr2;
    
    % check stopping criterion
    crit = relerr1<tol;
    if crit 
        nstall = nstall+1; 
    else
        nstall = 0; 
    end   
    if nstall>=3 || relerr2<tol 
        Out.exit = 1;
        break 
    end
    if toc(start_time)>maxT 
        Out.exit = 2;
        break 
    end   
end
fprintf('\n');
Out.iter = k;
end

function [obj, res]  = evaluation(Y, D, X, Corr, gamma)
res = norm(Corr.*(D*X)-Y, 'fro');
obj = 0.5*res^2 + gamma*sum(abs(X(:)));
end 