clear;
m = 36; K = 2*m; p = 20*m;
testnum = 1;

r = 6;
ratio = 0;

for num = 1:testnum    
    D = randn(m,K);
    for j=1:K
        D(:,j) = D(:,j)/norm(D(:,j));
    end
    X = []; Y = [];
    for j=1:p
        y = randn(r,1);
        id = randsample(K,r);
        x = D(:,id)*y;
        z = zeros(K,1);
        z(id) = y;
        Y = [Y,z]; % do not used. 
        X = [X,x];
    end
    
    D0 = randn(m,K);
    Y0 = randn(K,p);
    %%
    mu = 0.5/sqrt(m);
    opts.tol = 1e-4; opts.maxit = 1000;
    opts.D0 = D0; opts.Y0 = Y0;
    opts.yType = 0;
    t0 = tic;
    [Dl,Yl,Out] = dl_apg(X,K,mu,opts);
    timel = toc(t0);
    Dl = Dl*spdiags(1./sqrt(sum(Dl.^2))',0,K,K);
    identical_atoms = 0; epsilon = 1e-2;
    for i = 1:K
        atom = D(:,i);
        distances = 1-abs(atom'*Dl);
        mindist = min(distances);
        identical_atoms = identical_atoms + (mindist < epsilon);
    end
    identical_atoms/K
    ratio = ratio+identical_atoms/K;
end
ratio = ratio/testnum;
