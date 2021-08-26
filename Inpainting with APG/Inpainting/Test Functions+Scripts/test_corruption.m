clear all

testnum = 3;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record = zeros(testnum, 6, 13);
for i = 1:testnum
    Corr = rand(N,M);
    for k = 1:13
        corr = (Corr > (0.05*k+0.15));
        para.m = 49;
        para.blsize = 8;
        para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        data = [time, iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
        record(k,:,i) = data;
    end
end

result = sum(record,3)/testnum;


