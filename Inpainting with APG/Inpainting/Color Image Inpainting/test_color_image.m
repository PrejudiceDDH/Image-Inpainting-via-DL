clear all

testnum = 3;
img = imread('lena_color.jpg'); % uint8 format
[N,M] = size(img(:,:,1));

record = zeros(testnum,6,13);

para.m = 49;
para.blsize = 512;
para.blsize1 = 512; para.overlap1 = 128; 
para.blsize2 = 8; para.overlap2 = 4;  
para.gamma = 2/para.blsize2;

for i = 1:testnum
    Corr = rand(N,M); 
    % corr(3:6,:)=0; corr(77:80,:) =0; corr(250:253,:)=0; corr(480:483,:) = 0;
    % corr(:,45:46)=0; corr(:,360:363) = 0; corr(:,115:118)=0; corr(:, 170:173)=0;
    % corr(350:353,:) = 0;
    for k = 1:13
        corr = (Corr>(k*0.05+0.15));
        [~, ~, ~, qual_lasso1] = inpainting_color_sep(img, corr, para);
        [~, ~, ~, qual_lasso2] = inpainting_color_inter(img, corr, para, 1);
        [~, ~, ~, qual_lasso3] = inpainting_color_inter(img, corr, para, 0);
        data = [qual_lasso1(1),qual_lasso2(1),qual_lasso3(1),qual_lasso1(2),qual_lasso2(2),qual_lasso3(2)];
        record(i,:,k) = data;
    end
end

result = zeros(13, 6);
for i = 1:13
    result(i,:) = sum(record(:,:,i), 1)/testnum;
end


    
    

