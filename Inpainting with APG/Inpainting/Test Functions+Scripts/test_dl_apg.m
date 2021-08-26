clear;
n = 64; m = 100; p = 100*n;
testnum = 1;

r = 10;
ratio = 0;
redo = 0;

for num = 1:testnum    
    D = randn(n,m);
    Corr = rand(n,p); Corr = (Corr > 0.75);
    for j=1:m
        D(:,j) = D(:,j)/norm(D(:,j));
    end
    X = []; Y = [];
    for j=1:p
        x = randn(r,1); 
        ind = randsample(m,r);
        y = D(:,ind)*x;
        z = zeros(m,1);
        z(ind) = x;
        X = [X,z];
        Y = [Y,y];
    end
    Y = Corr.*Y;
    D0 = randn(n,m);
    X0 = randn(m,p);
    %%
    gamma = 2/sqrt(n);
    opts.tol = 1e-5; opts.maxit = 1000;
    opts.D0 = D0; opts.X0 = X0;
    
    t0 = tic;
    [Dl1,iter1, Out1, Xl1] = dl_apg(Y,Corr, m, gamma,opts);
    t1 = toc(t0);
    
    t0 = tic;
    [Dl2,iter2, Out2, Xl2] = dl_apg_c(Y,Corr, m, gamma,opts);
    t2 = toc(t0);
    
    t0 = tic;
    [Dl3, Out3, Xl3] = dl_apg_w_bb(Y,Corr, m, gamma,opts);
    t3 = toc(t0);
    
%     timel = toc(t0);
%     redo = redo + Out1.redoN;
%     Dl = Dl*spdiags(1./sqrt(sum(Dl.^2))',0,m,m);
%     identical_atoms = 0; epsilon = 1e-2;
%     for i = 1:m
%         atom = D(:,i);
%         distances = 1-abs(atom'*Dl);
%         mindist = min(distances);
%         identical_atoms = identical_atoms + (mindist < epsilon);
%     end
%     ratio = ratio+identical_atoms/m;
end

% ratio = ratio/testnum;