clear all

testnum = 1;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record = zeros(10, 10, testnum);

para.m = 49;
para.blsize = 8;
para.lasso = [0.2,0.4,0.6,0.8,1,1.5,2,2.5,3,3.5]/para.m^2;
GAMMA = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]/para.blsize;

for i = 1:testnum
    corr = rand(N, M); corr = (corr > 0.75); 
    for k = 1:10
        para.gamma = GAMMA(k);
        [qual_lasso, ~] = testfunc_lasso(img, corr, para);
        record(k,:,i) = qual_lasso;
    end
end

result = sum(record,3)/testnum;




