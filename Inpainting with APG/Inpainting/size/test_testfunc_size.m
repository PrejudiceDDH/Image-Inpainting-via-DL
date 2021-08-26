clear all

testnum = 3;
img = imread('man.pgm'); %img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record = zeros(testnum, 6, 8);
for i = 1:testnum
    corr = rand(N, M); corr = (corr > 0.75); 
    for k = 1:8
        fprintf('Testnum: %i, 8/4, %i \n', i, k);
        img_p = img(1:128*k,1:128*k); corr_p = corr(1:128*k,1:128*k);
        para.m = 49;
        para.blsize = 8;
        para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img_p, corr_p, para);
        record(k,:,i) = [time, iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end
end

result = sum(record,3)/testnum;

