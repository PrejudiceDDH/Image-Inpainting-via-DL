function [time, iteration, qual_lasso] = testfunc_c(img, corr, para)
img_corr = corr.*img;

% % temporarily
% [N,M] = size(img);
% blsize2 = para.blsize2;
% exceed = blsize2/2 - mod(512,blsize2/2);
% img = img(1:N-exceed, 1:M-exceed);
% img_corr = img_corr(1:N-exceed, 1:M-exceed);
% corr = corr(1:N-exceed, 1:M-exceed);
% para.blsize = N-exceed; para.blsize1 = N-exceed;

[img_lasso, iteration, time] = inpainting_single_c(img_corr, corr, para);
% img_ormp = img_ormp; img_lasso = img_lasso;
imshow(img_lasso)
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end

function [img_lasso, iteration, time] = inpainting_single_c(img_corr, corr, para)
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
t0 = tic;
[Dl, iteration] = dl_apg(Y, Corr, m, gamma, opts);

% Sparse Coding with ORMP/ Least Square.
X_lasso = zeros(m,p);
for l = 1:p
    X_lasso(:,l) = lasso(diag(Corr(:,l))*Dl, Y(:,l), 'lambda', 0.5/(m^2));
end
time = toc(t0);

recovered_lasso = Dl*X_lasso;
img_lasso = patch2img_copy(recovered_lasso, blsize, overlap, [N, M]);
end