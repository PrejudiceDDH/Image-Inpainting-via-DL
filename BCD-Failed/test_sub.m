D = randn(10,40);
for j=1:40
    D(:,j) = D(:,j)/norm(D(:,j));
end
Dt = D';
Dsq = Dt*D;
Y = randn(10,100);
X = randn(40,100);
Lx = 1;
gamma = 0.5/sqrt(10);
opts = struct([]);
[Z, Out] = dl_sub(Y,X,Dt, Dsq, Lx, gamma, opts)
