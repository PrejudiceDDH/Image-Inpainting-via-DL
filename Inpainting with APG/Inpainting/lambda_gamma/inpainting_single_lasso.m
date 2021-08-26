function [img_lasso, time_lasso, iteration, time] = inpainting_single(img_corr, corr, para)
% para: m, blsize (overlap = 50% by default)
% img_corr: grayscale corrupted image.
% Output: img_lasso: N*M*num
% [N0, M0] = size(img_corr); 
% para.lasso: to control the lasso parameter - is a vector

num = size(para.lasso,2); LAMBDA = para.lasso;

m = para.m; gamma = para.gamma; 
blsize = para.blsize; overlap = blsize/2;
[N,M] = size(img_corr);

DCT=zeros(blsize,sqrt(m));
for l=0:1:sqrt(m)-1
    V=cos([0:1:blsize-1]'*l*pi/sqrt(m));
    if l>0 
        V=V-mean(V); 
    end
    DCT(:,l+1)=V/norm(V);
end 
D0=kron(DCT,DCT); 

Y = patch_extract(img_corr, blsize, overlap); [n,p] = size(Y);
Corr = patch_extract(corr, blsize, overlap);

X0 = randn(m,p);
opts.tol = 1e-4; opts.maxit = 10000;
opts.D0 = D0; opts.X0 = X0;        

% Dictionary learning, initialize with the DCT dictionary.
% 0.5 seems to be better than 1
t0 = tic;
[Dl, iteration] = dl_apg(Y, Corr, m, gamma, opts);
time = toc(t0);

% Sparse Coding with ORMP/ Least Square.
img_lasso = zeros(N,M,num);
time_lasso = zeros(1,num);
for k = 1:num
    X_lasso = zeros(m,p); 
    tt = tic;
	for l = 1:p
	    X_lasso(:,l) = lasso(diag(Corr(:,l))*Dl, Y(:,l), 'lambda', LAMBDA(k));
	end
	time_lasso(k) = toc(tt);
	recovered_lasso = Dl*X_lasso;
	img_lasso(:,:,k) = patch2img_copy(recovered_lasso, blsize, overlap, [N, M]);
end
end