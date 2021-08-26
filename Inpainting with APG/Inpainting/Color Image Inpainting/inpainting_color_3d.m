function [time, iteration, qual_ormp, qual_lasso] = inpainting_color_3d(img, corr, para)
% img: N*M*3; corr: N*M
% formulate the color img as a long img (not long patch)
img = im2double(img); % convey to the double precision
for i = 1:3
	img_corr(:,:,i) = corr.*img(:,:,i);
end
[img_ormp, img_lasso, iteration, time] = inpainting_3d(img_corr, corr, para);
imshow(img_ormp)
qual_ormp(1) = psnr(img, img_ormp);
qual_ormp(2) = ssim(img, img_ormp);
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end

function [img_ormp, img_lasso, iteration, time] = inpainting_3d(img_corr, corr, para)
% para: m, blsize (overlap = 50% by default)
% img_corr: grayscale corrupted image.
% Output: img_ormp, img_lasso

% [N0, M0] = size(img_corr); 
m = para.m; gamma = para.gamma; % by default, should be 2/blsize
blsize = para.blsize; overlap = blsize/2;
[N,M,~] = size(img_corr);

DCT=zeros(blsize,sqrt(m));
for l=0:1:sqrt(m)-1
    V=cos([0:1:blsize-1]'*l*pi/sqrt(m));
    if l>0 
        V=V-mean(V); 
    end
    DCT(:,l+1)=V/norm(V);
end 
D0=kron(DCT,DCT); 

for i = 1:3
	Y(:,:,i) = patch_extract(img_corr(:,:,i), blsize, overlap); 
end
p = size(Y, 2);
Corr = patch_extract(corr, blsize, overlap);

X0 = randn(m,p);
opts.tol = 1e-4; opts.maxit = 10000;
opts.D0 = D0; opts.X0 = X0;        

% Dictionary learning, initialize with the DCT dictionary.
% 0.5 seems to be better than 1
t0 = tic;
[Dl, iteration] = dl_apg_3d(Y, Corr, m, gamma, opts);
time = toc(t0);
D1 = Dl(:,:,1); D2 = Dl(:,:,2); D3 = Dl(:,:,3);

% Sparse Coding with ORMP/ Least Square.
X_ormp = zeros(m, p); X_lasso = zeros(m,p);
op = struct('targetNonzeros', blsize, 'verbose', 1);
for l = 1:p
	Mask = diag(Corr(:,l));
	S = D1'*Mask*D1+D2'*Mask*D2+D3'*Mask*D3;
	[U,Sigma,~] = svd(S); 
	C = U*sqrt(Sigma)*U'; C_inv = U*pinv(sqrt(Sigma))*U';
	b = D1'*Mask*Y(:,l,1)+D2'*Mask*Y(:,l,2)+D3'*Mask*Y(:,l,3);
    X_ormp(:,l) = sparseapprox(C_inv*b, C, 'ORMP', op);
    X_lasso(:,l) = lasso(C, C_inv*b, 'lambda', 1/(m^2));
end
img_ormp = zeros(N,M,3); img_lasso = img_ormp;
for i = 1:3
    img_ormp(:,:,i) = patch2img_copy(Dl(:,:,i)*X_ormp, blsize, overlap, [N, M]);
	img_lasso(:,:,i) = patch2img_copy(Dl(:,:,i)*X_lasso, blsize, overlap, [N, M]);
end
end