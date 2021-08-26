function [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para)
img_corr = corr.*img;

% % temporarily
% [N,M] = size(img);
% blsize2 = para.blsize2;
% exceed = blsize2/2 - mod(512,blsize2/2);
% img = img(1:N-exceed, 1:M-exceed);
% img_corr = img_corr(1:N-exceed, 1:M-exceed);
% corr = corr(1:N-exceed, 1:M-exceed);
% para.blsize = N-exceed; para.blsize1 = N-exceed;

[img_ormp, img_lasso, iteration, time] = inpainting_single(img_corr, corr, para);
% img_ormp = img_ormp; img_lasso = img_lasso;

qual_ormp(1) = psnr(img, img_ormp);
qual_ormp(2) = ssim(img, img_ormp);
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end