clear all

testnum = 1;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

para1.m = 49;
para1.blsize = 8;
para1.gamma = 2/para1.blsize;

para2.bb = 8; 
para2.overlap = 4; 
para2.J = 5; 
para2.m = 49;

para3.bb = 8; 
para3.overlap = 4; 
para3.J = 5; 
para3.m = 256;

para4.bb = 8; 
para4.overlap = 4; 
para4.J = 1; 
para4.m = 49;

record = zeros(13, 12, testnum);
for i = 1:testnum
    Corr = rand(N, M); 
    for k = 12:12
        fprintf('Now k equals %2i \n', k);
        corr = (Corr > (k*0.05+0.15)); 
%         [time1, ~, qual_lasso] = testfunc_c(img, corr, para1);
%         [time2, qual2, ~] = inpainting_ksvd(img, corr, para2);
%         [time3, qual3, ~] = inpainting_ksvd(img, corr, para3);
        [time4, qual4, ~] = inpainting_ksvd(img, corr, para4);
        data = [time1, time2, time3, time4, qual_lasso(1), qual2(1), qual3(1), qual4(1), qual_lasso(2), qual2(2), qual3(2), qual4(2)];
        record(k,:,i) = data;
    end
end

result = sum(record,3)/testnum;


