function [time, iteration, qual_ormp, qual_lasso] = inpainting_color_mix(img, corr, para)
% img: N*M*3; corr: N*M
% formulate the color img as a long img (not long patch)
img = im2double(img); % convey to the double precision
for j = 1:3
	img_corr(:,:,j) = corr.*img(:,:,j);
end

[img_ormp, img_lasso, iteration, time] = inpainting_mix(img_corr, corr, para);

qual_ormp(1) = psnr(img, img_ormp);
qual_ormp(2) = ssim(img, img_ormp);
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end



function [img_ormp, img_lasso, iteration, time] = inpainting_mix(img_corr, corr, para)
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

n_row = (N-overlap)/(blsize-overlap);
n_col = (M-overlap)/(blsize-overlap);
p = n_row*n_col*3;
Y = zeros(blsize^2, p); Corr = Y;
for i = 1:3
	Y(:,i:3:p-3+i) = patch_extract(img_corr(:,:,i), blsize, overlap); 
	Corr(:,i:3:p-3+i) = patch_extract(corr, blsize, overlap);
end

X0 = randn(m,p);
opts.tol = 1e-4; opts.maxit = 10000;
opts.D0 = D0; opts.X0 = X0;        

% Dictionary learning, initialize with the DCT dictionary.
% 0.5 seems to be better than 1
t0 = tic;
[Dl, iteration] = dl_apg(Y, Corr, m, gamma, opts);
time = toc(t0);

% Sparse Coding with ORMP/ Least Square.
X_ormp = zeros(m, p); X_lasso = zeros(m,p);
op = struct('targetNonzeros', blsize, 'verbose', 1);
for l = 1:p
    X_ormp(:,l) = sparseapprox(Y(:,l), diag(Corr(:,l))*Dl, 'ORMP', op);
    X_lasso(:,l) = lasso(diag(Corr(:,l))*Dl, Y(:,l), 'lambda', 1/(m^2));
end

recovered_ormp = Dl*X_ormp; recovered_lasso = Dl*X_lasso;
for i = 1:3
	img_ormp(:,:,i) = patch2img_copy(recovered_ormp(:,i:3:p-3+i), blsize, overlap, [N, M]);
	img_lasso(:,:,i) = patch2img_copy(recovered_lasso(:,i:3:p-3+i), blsize, overlap, [N, M]);
end
end