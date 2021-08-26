clear;
n = 36; m = 2*n; p = 20*n;
testnum = 1;

r = 6; % Control the Sparsity
ratio = 0;

for num = 1:testnum    
    D = randn(n,m);
%     for j=1:m
%         D(:,j) = D(:,j)/norm(D(:,j));
%     end
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
    gamma = 1;%/sqrt(n);
    opts.tol = 1e-4; opts.maxit = 10000;
    opts.D0 = D0; opts.X0 = X0;
    t0 = tic;
    [Dl,Xl,Out] = dl_abcd(Y,m,gamma,opts);
    timel = toc(t0);
%     identical_atoms = 0; epsilon = 1e-2;
%     Dist = zeros(m, 1);
%     for i = 1:m
%         atom = D(:,i);
%         distances = 1-abs(atom'*Dl);
%         mindist = min(distances);
%         Dist(i) = mindist;
%         identical_atoms = identical_atoms + (mindist < epsilon);
%     end
%     ratio = ratio+identical_atoms/m;    
end
% ratio = ratio/testnum;