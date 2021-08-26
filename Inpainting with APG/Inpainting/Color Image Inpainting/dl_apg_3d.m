function [D, iteration, Out, X] = dl_apg_3d(Y, Corr, m, gamma, opts)
% Y a 3-dim matrix, corresponds to the corrupted image
% Corr is the 2-dim mask matrix
% consists of 0/1 entries, 0 for masked pixels.
[n,p,~] = size(Y); % [m, p] = size(X), 

if isfield(opts,'tol')       tol = opts.tol;      else tol = 1e-5;     end
if isfield(opts,'maxit')     maxit = opts.maxit;  else maxit = 500;    end
if isfield(opts,'maxT')      maxT = opts.maxT;    else maxT = 1e5;     end

if isfield(opts,'D0')        D0 = opts.D0;        else D0 = randn(n,m);end
if isfield(opts,'X0')        X0 = opts.X0;        else X0 = randn(m,p);end

% normalize D0, here D0 is still a 2-dim matrix.
for j=1:m
    D0(:,j) = D0(:,j)/norm(D0(:,j));
end
for i = 1:3
    D(:,:,i) = D0;
end

% computing the projection matrix.
M = zeros(size(Corr,1), size(Corr,1), p);
for i=1:p
    M(:,:,i) = diag(Corr(:,i));
end
nrmY = sqrt(sum((Y(:)).^2));

Dm = D; Xm = X0; LipX = norm(D(:,:,1))^2 + norm(D(:,:,2))^2 + norm(D(:,:,3))^2;
t0 = 1; Lx = 1; Ld = 1;
rw = 0.9999;

obj0 = evaluation(Y, D, X0, Corr, gamma);
obj = obj0;

Out.redoN = 0; nstall_1 = 0; start_time = tic;
fprintf('Iteration:     ');

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k);
    
    % update X
    % in BCD, we care about the Lipschitz constant because it ensures
    % the convergence of the proximal gradient algorithm in subproblem
    Lx0 = Lx;       Lx = LipX/2; 
    GradX = zeros(m,p);
    for i=1:p
        GradX(:,i) = D(:,:,1)'*(M(:,:,i)*D(:,:,1)*Xm(:,i)-M(:,:,i)*Y(:,i,1))+D(:,:,2)'*(M(:,:,i)*D(:,:,2)*Xm(:,i)-M(:,:,i)*Y(:,i,2))+D(:,:,3)'*(M(:,:,i)*D(:,:,3)*Xm(:,i)-M(:,:,i)*Y(:,i,3));
    end
    X = Xm-GradX/Lx;
    X = sign(X).*max(abs(X)-gamma/Lx,0);
    
    % update D
    Ld0 = Ld;       Ld = norm(X)^2/2;      
    GradD = zeros(n,m,3);
    for i = 1:p
        for j = 1:3
            GradD(:,:,j) = GradD(:,:,j) + M(:,:,i)*(Dm(:,:,j)*X(:,i)-Y(:,i,j))*X(:,i)';
        end
    end
    D = Dm-GradD/Ld;
    for j = 1:3
        nrmD = sqrt(sum(D(:,:,j).^2,1)); 
        id = nrmD>1; s = sum(id);
        if s>0     
            D(:,id,j) = D(:,id,j)*spdiags(nrmD(id)'.^(-1),0,s,s);
        end
    end
    LipX0 = LipX;
    LipX = norm(D(:,:,1))^2 + norm(D(:,:,2))^2 + norm(D(:,:,3))^2;

    
    obj0 = obj;
    [obj, res] = evaluation(Y, D, X, Corr, gamma);
   
    if obj>obj0 % Then, we do not employ the acceleration
        Out.redoN = Out.redoN+1;
        
        % update X
        Lx = LipX0; % last step data
        for i=1:p
            GradX(:,i) = D0(:,:,1)'*(M(:,:,i)*D0(:,:,1)*X0(:,i)-M(:,:,i)*Y(:,i,1))+D0(:,:,2)'*(M(:,:,i)*D0(:,:,2)*X0(:,i)-M(:,:,i)*Y(:,i,2))+D0(:,:,3)'*(M(:,:,i)*D0(:,:,3)*X0(:,i)-M(:,:,i)*Y(:,i,3));
        end
        X = X0-GradX/Lx;
        X = sign(X).*max(abs(X)-gamma/Lx,0);
        
        
        % update D
        Ld = norm(X)^2;
        GradD = zeros(n,m,3);
        for i = 1:p
            for j = 1:3
                GradD(:,:,j) = GradD(:,:,j) + M(:,:,i)*(D0(:,:,j)*X(:,i)-Y(:,i,j))*X(:,i)';
            end
        end
        D = D0-GradD/Ld;
        for j = 1:3
            nrmD = sqrt(sum(D(:,:,j).^2,1)); 
            id = nrmD>1; s = sum(id);
            if s>0     
                D(:,id,j) = D(:,id,j)*spdiags(nrmD(id)'.^(-1),0,s,s);
            end
        end
        
        obj0 = obj; 
        [obj, res] = evaluation(Y,D,X,Corr,gamma);

        Out.D{k} = D;
    end
    % do extrapolation
    t = (1+sqrt(1+4*t0^2))/2;
    w = (t0-1)/t; % extrapolation weight
    % w = 0;
    wX = min([w,rw*sqrt(Lx0/Lx)]);
    wD = min([w,rw*sqrt(Ld0/Ld)]);
    Xm = X+wX*(X-X0);   Dm = D+wD*(D-D0);
    t0 = t; X0 = X; D0 = D;
    
    % --- diagnostics, reporting, stopping checks ---
    relerr1 = abs(obj-obj0)/(obj0+1);    relerr2 = res/nrmY;
    relerr3 = sqrt(sum((GradX(:)).^2))/Lx + sqrt(sum((GradD(:)).^2))/Ld;
    
    % reporting
    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1;
    Out.hist_rel(2,k) = relerr2;
    Out.hist_rel(3,k) = relerr3;
    
    % check stopping criterion
    crit_1 = relerr1<tol;
    if crit_1 
        nstall_1 = nstall_1+1; 
    else
        nstall_1 = 0; 
    end
    
    crit_3 = relerr3<10^(-6);
    if crit_3 
        nstall_3 = nstall_3+1; 
    else
        nstall_3 = 0; 
    end
    
    if nstall_1>=3
        Out.exit = 1;
        break 
    elseif relerr2<tol
        Out.exit = 2;
        break
    elseif nstall_3 >=3
        Out.exit = 3;
        break
    elseif toc(start_time)>maxT
        Out.exit = 4;
        break
    end
end

Lx0 = Lx; Lx = LipX/2; 
GradX = zeros(m,p);
for i=1:p
    GradX(:,i) = D(:,:,1)'*(M(:,:,i)*D(:,:,1)*Xm(:,i)-M(:,:,i)*Y(:,i,1))+D(:,:,2)'*(M(:,:,i)*D(:,:,2)*Xm(:,i)-M(:,:,i)*Y(:,i,2))+D(:,:,3)'*(M(:,:,i)*D(:,:,3)*Xm(:,i)-M(:,:,i)*Y(:,i,3));
end
X = Xm-GradX/Lx;
X = sign(X).*max(abs(X)-gamma/Lx,0);

fprintf('\n');
iteration = k;
end

function [obj, res]  = evaluation(Y, D, X, Corr, gamma)
res = norm(Corr.*(D(:,:,1)*X)-Y(:,:,1), 'fro')+norm(Corr.*(D(:,:,2)*X)-Y(:,:,2), 'fro')+norm(Corr.*(D(:,:,3)*X)-Y(:,:,3), 'fro');
obj = 0.5*res^2 + gamma*sum(abs(X(:)));
end 