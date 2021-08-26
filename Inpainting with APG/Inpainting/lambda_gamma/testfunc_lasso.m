function [qual_lasso, time_lasso] = testfunc_lasso(img, corr, para)

img_corr = corr.*img;
num = size(para.lasso,2);
[img_lasso, time_lasso, ~, ~] = inpainting_single_lasso(img_corr, corr, para);
for i = 1:num
	qual_lasso(i) = psnr(img, img_lasso(:,:,i));
end
end