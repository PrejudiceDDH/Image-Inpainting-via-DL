clear all

testnum = 3;
img = imread('lena_color.jpg'); % uint8 format
[N,M,~] = size(img);

% record = zeros(testnum,36,13);
record1 = zeros(13,6,testnum);
record2 = zeros(13,12,testnum);

para.m = 49;
para.blsize = 8;
para.gamma = 2/para.blsize;

for i = 1:testnum
    Corr = rand(N, M); %Corr = (Corr > 0.75); 
    for k = 1:13
        fprintf('Now k equals %2i \n', k);
        corr = (Corr>(k*0.05+0.15));
        [time1, iteration1, qual_ormp1, qual_lasso1] = inpainting_color_sep(img, corr, para);
        [time2, iteration2, qual_ormp2, qual_lasso2] = inpainting_color_inter(img, corr, para, 1);
        [time3, iteration3, qual_ormp3, qual_lasso3] = inpainting_color_inter(img, corr, para, 0);
%         [time4, iteration4, qual_ormp4, qual_lasso4] = inpainting_color_mix(img, corr, para);
%         [time5, iteration5, qual_ormp5, qual_lasso5] = inpainting_color_long(img, corr, para);
%         [time6, iteration6, qual_ormp6, qual_lasso6] = inpainting_color_3d(img, corr, para);
%         data1 = [qual_lasso1(1),qual_lasso2(1),qual_lasso3(1),qual_lasso4(1),qual_lasso5(1),qual_lasso6(1), qual_lasso1(2),qual_lasso2(2),qual_lasso3(2), qual_lasso4(2),qual_lasso5(2),qual_lasso6(2)];
%         data2 = [qual_ormp1(1),qual_ormp2(1),qual_ormp3(1),qual_ormp4(1),qual_ormp5(1),qual_ormp6(1), qual_ormp1(2),qual_ormp2(2),qual_ormp3(2), qual_ormp4(2),qual_ormp5(2),qual_ormp6(2)];
%         data3 = [time1, time2, time3, time4, time5, time6, iteration1, iteration2, iteration3, iteration4, iteration5, iteration6];
        time = [time1,time2,time3];
        iteration = [iteration1, iteration2, iteration3];
        qual1 = [qual_lasso1(1),qual_lasso2(1),qual_lasso3(1), qual_lasso1(2),qual_lasso2(2),qual_lasso3(2)];
        qual2 = [qual_ormp1(1),qual_ormp2(1),qual_ormp3(1), qual_ormp1(2),qual_ormp2(2),qual_ormp3(2)];
        record1(k,:,i) = [time,iteration];
        record2(k,:,i) = [qual1,qual2];
    end
end

result1 = sum(record1,3)/testnum;
result2 = sum(record2,3)/testnum;



    
    

