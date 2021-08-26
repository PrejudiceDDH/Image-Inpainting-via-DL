function [time, iteration, qual_ormp, qual_lasso] = inpainting_color_long(img, corr, para)
% img: N*M*3; corr: N*M
% formulate the color img as a long img (not long patch)
img = im2double(img); % convey to the double precision
M = size(img,2);
img_corr = [corr.*img(:,:,1), corr.*img(:,:,2), corr.*img(:,:,3)];
Corr = [corr,corr,corr];
[img_ormp0, img_lasso0, iteration, time] = inpainting_single(img_corr, Corr, para);
img_ormp = zeros(size(img)); img_lasso = img_ormp;
for i = 1:3
    img_ormp(:,:,i) = img_ormp0(:,((i-1)*M+1):i*M);
    img_lasso(:,:,i) = img_lasso0(:,((i-1)*M+1):i*M);
end

qual_ormp(1) = psnr(img, img_ormp);
qual_ormp(2) = ssim(img, img_ormp);
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end