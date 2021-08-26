function [img_ormp, img_lasso, img_apg] = inpainting_single_apg(img_corr, corr, para)
% para: m, blsize (overlap = 50% by default)
% img_corr: grayscale corrupted image.
% Output: img_ormp, img_lasso

% [N0, M0] = size(img_corr); 
m = para.m; gamma = para.gamma; % by default, should be 2/blsize
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

Y = patch_extract(img_corr, blsize, overlap); p = size(Y, 2);
Corr = patch_extract(corr, blsize, overlap);

X0 = randn(m,p);
opts.tol = 1e-4; opts.maxit = 10000;
opts.D0 = D0; opts.X0 = X0;        

% Dictionary learning, initialize with the DCT dictionary.
% 0.5 seems to be better than 1
[Dl, ~, ~, X_apg] = dl_apg(Y, Corr, m, gamma, opts);

% Sparse Coding with ORMP/ Least Square.
X_ormp = zeros(m, p); X_lasso = zeros(m,p); 
op = struct('targetNonzeros', blsize, 'verbose', 1);
for l = 1:p
    X_ormp(:,l) = sparseapprox(Y(:,l), diag(Corr(:,l))*Dl, 'ORMP', op);
    X_lasso(:,l) = lasso(diag(Corr(:,l))*Dl, Y(:,l), 'lambda', 0.5/(m^2));
end

recovered_ormp = Dl*X_ormp; recovered_lasso = Dl*X_lasso; recovered_apg = Dl*X_apg;
img_ormp = patch2img_copy(recovered_ormp, blsize, overlap, [N, M]);
img_lasso = patch2img_copy(recovered_lasso, blsize, overlap, [N, M]);
img_apg = patch2img_copy(recovered_apg, blsize, overlap, [N, M]);
end