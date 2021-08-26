clear all

testnum = 1;

img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);
qual_ormp = zeros(2,testnum); qual_lasso = zeros(2,testnum);
time = zeros(1,testnum);

para.m = 49;
blsize = 512;
para.blsize1 = 512; para.overlap1 = 128; 
para.blsize2 = 10; para.overlap2 = 5; para.gamma = 2/para.blsize2;

% temporarily
blsize2 = para.blsize2;
exceed = blsize2/2 - mod(512,blsize2/2);
img = img(1:N-exceed, 1:M-exceed);
% img_corr = img_corr(1:N-exceed, 1:M-exceed);
% corr = corr(1:N-exceed, 1:M-exceed);

for i = 1:testnum
    corr = rand(N-exceed, M-exceed); corr = (corr > 0.75); 
    img_corr = corr.*img;
    
    t0 = tic;
    [img_ormp, img_lasso, iteration, time] = inpainting_triple(img_corr, corr, para, blsize);
    time(i) = toc(t0);

    qual_ormp(1,i) = psnr(img, img_ormp);
    qual_ormp(2,i) = ssim(img, img_ormp);
    qual_lasso(1,i) = psnr(img, img_lasso);
    qual_lasso(2,i) = ssim(img, img_lasso);
end



