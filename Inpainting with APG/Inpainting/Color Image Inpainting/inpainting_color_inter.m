function [time, iteration, qual_ormp, qual_lasso] = inpainting_color_inter(img, corr, para, Type)
% img: N*M*3, uin8 format; corr: N*M
img_c = rgb2ycbcr(img);
for k = 1:3
    img_c(:,:,k) = corr.*double(img_c(:,:,k));
end
img_inpaint = zeros(size(img_c));
img_inpaint(:,:,2) = Interpolation(double(img_c(:,:,2)),~corr); 
img_inpaint(:,:,3) = Interpolation(double(img_c(:,:,3)),~corr);
switch Type
	case 0
		img_inpaint(:,:,1) = Interpolation(double(img_c(:,:,1)),~corr); 
		img_inpaint = ycbcr2rgb(uint8(img_inpaint));

        time = 0; iteration = 0;
		qual_lasso(1) = psnr(img, img_inpaint);
		qual_lasso(2) = ssim(img, img_inpaint);
		qual_ormp(1) = qual_lasso(1);
		qual_ormp(2) = qual_lasso(2);
	case 1
		img_corr = im2double(img_c(:,:,1));
		[img_inpaint_ormp, img_inpaint_lasso, iteration, time] = inpainting_single(img_corr, corr, para);
 		img_inpaint_ormp = 255*img_inpaint_ormp;
 		img_inpaint_lasso = 255*img_inpaint_lasso;

		img_inpaint(:,:,1) = img_inpaint_ormp;
		img_ormp = ycbcr2rgb(uint8(img_inpaint));
		img_inpaint(:,:,1) = img_inpaint_lasso;
		img_lasso = ycbcr2rgb(uint8(img_inpaint));
 		
		qual_ormp(1) = psnr(img, img_ormp);
		qual_ormp(2) = ssim(img, img_ormp);
		qual_lasso(1) = psnr(img, img_lasso);
		qual_lasso(2) = ssim(img, img_lasso);
end
end