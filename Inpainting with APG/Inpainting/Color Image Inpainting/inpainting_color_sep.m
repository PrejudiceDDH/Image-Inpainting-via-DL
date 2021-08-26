function [time, iteration, qual_ormp, qual_lasso] = inpainting_color_sep(img, corr, para)
% img: N*M*3; corr: N*M
img = im2double(img); % convery to the double precision
time = 0; iteration = 0;
for j = 1:3
	img_corr(:,:,j) = corr.*img(:,:,j);
	[img_ormp(:,:,j), img_lasso(:,:,j), iter, t] = inpainting_single(img_corr(:,:,j), corr, para);
	time = time+t; iteration = iteration +iter;
end
qual_ormp(1) = psnr(img, img_ormp);
qual_ormp(2) = ssim(img, img_ormp);
qual_lasso(1) = psnr(img, img_lasso);
qual_lasso(2) = ssim(img, img_lasso);
end