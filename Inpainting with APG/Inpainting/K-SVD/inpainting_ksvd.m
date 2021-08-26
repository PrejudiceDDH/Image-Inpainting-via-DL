function [time, qual, img_out] = inpainting_ksvd(img, corr, para)
% para: b, m, overlap, J
% wrote for grayscale image
% img_corr is a 2D matrix, double precision
img_corr = img.*corr;
bb = para.bb; overlap = para.overlap; J = para.J; m = para.m;
[N,M]=size(img_corr);
NN = ceil((N-bb)/overlap) * overlap + bb;
MM = ceil((M-bb)/overlap) * overlap + bb;
img_new = zeros(NN,MM);
img_new(1:N,1:M) = img_corr;

%Compute mask and extracting its patches
Mask = double(~(img_new==0));
blkMask = overlap_im2col(Mask, bb, overlap);

% The dictionary DCT
DCT=zeros(bb,sqrt(m));
for k=0:1:sqrt(m)-1,
    V=cos([0:1:bb-1]'*k*pi/sqrt(m));
    if k>0 
        V=V-mean(V); 
    end;
    DCT(:,k+1)=V/norm(V);
end;
DCT=kron(DCT,DCT);

% Extracting the noisy image patches
blkMatrixIm = overlap_im2col(img_new, bb, overlap);

sigma = 0.01;  % 0.005 0.01
rc_min = 0.01; % rc_min: minimal residual correlation before stopping pursuit
max_coeff = 20; % max_coeff: sparsity constraint for signal representation 10

% Inpainting the Patches (m-SVD)
t0 = tic;
[Dict, Coeff]=KSVD_Inpainting(DCT,blkMatrixIm,blkMask,sigma,rc_min,max_coeff,J);  %DCT
time = toc(t0);

% Creating the output image
img_out = overlap_col2im(Dict*Coeff, blkMask, bb, overlap, size(img_new));
img_out=max(min(img_out,1),0);
imshow(img_out);
qual(1) = psnr(img, img_out);
qual(2) = ssim(img, img_out);
end