clear all
clc

testnum = 3;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record1 = zeros(testnum, 6, 13);
% record2 = zeros(testnum, 6, 13);
for i = 1:testnum
    Corr = rand(N, M);
    for k = 1:13
        corr = (Corr > (0.05*k+0.15)); 
        fprintf('Testnum: %i, 8/4, %i \n', i, k);
        para.m = 49; para.blsize = 8; para.gamma = 1/256;
        [qual_ormp, qual_lasso, qual_apg] = testfunc_apg(img, corr, para);
        data = [qual_lasso(1), qual_ormp(1), qual_apg(1), qual_lasso(2), qual_ormp(2), qual_apg(2)];
        record1(i,:,k) = data;
    end
%     for k = 1:13
%         corr = (Corr > (0.05*k+0.15)); 
%         fprintf('Testnum: %i, 8/4, %i \n', i, k);
%         para.m = 49; para.blsize = 8; para.gamma = 1/64;
%         [qual_ormp, qual_lasso, qual_apg] = testfunc_apg(img, corr, para);
%         data = [qual_lasso(1), qual_ormp(1), qual_apg(1), qual_lasso(2), qual_ormp(2), qual_apg(2)];
%         record2(i,:,k) = data;
%     end
end

result1 = zeros(13, 6);
for i = 1:13
    result1(i,:) = sum(record1(:,:,i), 1)/testnum;
end

% result2 = zeros(13, 6);
% for i = 1:13
%     result2(i,:) = sum(record2(:,:,i), 1)/testnum;
% end
