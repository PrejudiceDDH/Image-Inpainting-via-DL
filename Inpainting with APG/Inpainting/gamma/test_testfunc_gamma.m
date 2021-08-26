clear all

testnum = 3;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record = zeros(15, 5, testnum);

para.m = 49;
para.blsize = 8;
GAMMA = [0.25, 0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,8,10]/para.blsize;

for i = 1:testnum
    corr = rand(N, M); corr = (corr > 0.75); 
    for k = 1:15
        para.gamma = GAMMA(k);
        [time, ~, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record(k,:,i) = [time,qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end
end

result = sum(record,3)/testnum;




