clear;
n = 36; m = 2*n; p = 20*n;
testnum = 20;

r = 6; % Control the Sparsity
ratio = 0;

for num = 1:testnum
    ratio = 0;
    D = randn(n,m);
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
    
    D0 = randn(n,m);
    X0 = randn(m,p);
    %%
    gamma = 0.5/sqrt(n);
    opts.tol = 1e-4; opts.maxit = 10000;
    opts.D0 = D0; opts.X0 = X0;
    t0 = tic;
    [Dl,Xl,Out] = dl_bcd(Y,m,gamma,opts);
    timel = toc(t0);
    identical_atoms = 0; epsilon = 1e-2;
    for i = 1:m
        atom = D(:,i);
        distances = 1-abs(atom'*Dl);
        mindist = min(distances);
        identical_atoms = identical_atoms + (mindist < epsilon);
    end
    ratio = ratio+identical_atoms/m;    
    Record.D{num} = D; Record.ratio{num} = ratio;
    Record.Y{num} = Y; Record.Dl{num} = Dl;
    Record.Xl{num} = Xl; Record.iter{num} = Out.iter;
end
%ratio = ratio/testnum;
