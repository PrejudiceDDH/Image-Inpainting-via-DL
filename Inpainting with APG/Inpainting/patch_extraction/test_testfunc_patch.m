clear all

testnum = 3;
img = imread('lena_color.jpg'); img = rgb2gray(img); 
img = im2double(img); [N,M] = size(img);

record1 = zeros(12, 6, testnum);
record2 = zeros(10, 6, testnum);
record3 = zeros(10, 6, testnum);
record4 = zeros(10, 6, testnum);
record5 = zeros(10, 6, testnum);
record6 = zeros(5, 6, testnum);

for i = 1:testnum
    corr = rand(N, M); corr = (corr > 0.75); 
    for k = 1:12
    	para.m = (k+4)^2; 
    	para.blsize = 8;
    	para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record1(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end

    for k = 1:10
    	para.m = (k+6)^2; 
    	para.blsize = 10;
    	para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record2(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end

    for k = 1:10
    	para.m = (k+6)^2; 
    	para.blsize = 12;
    	para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record3(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end

    for k = 1:10
    	para.m = (k+8)^2; 
    	para.blsize = 14;
    	para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record4(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end

    for k = 1:10
    	para.m = (k+10)^2; 
    	para.blsize = 16;
    	para.gamma = 2/para.blsize;
        [time, iteration, qual_ormp, qual_lasso] = testfunc(img, corr, para);
        record5(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end

    for k = 1:5
    	para.m = (k+4)^2;
    	para.blsize1 = 256; para.blsize2 = 8; para.gamma = 2/para.blsize2;
    	[time, iteration, qual_ormp, qual_lasso] = testfunc_dual(img, corr, para);
    	record6(k,:,i) = [time,iteration, qual_lasso(1), qual_ormp(1), qual_lasso(2), qual_ormp(2)];
    end
end

result1 = sum(record1,3)/testnum;
result2 = sum(record2,3)/testnum;
result3 = sum(record3,3)/testnum;
result4 = sum(record4,3)/testnum;
result5 = sum(record5,3)/testnum;
result6 = sum(record6,3)/testnum;




