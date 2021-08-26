function [img_ormp, img_lasso, iteration, time] = inpainting_dual(img_corr, corr, para)
% para: m, blsize1, blsize2, gamma
% img_corr: grayscale corrupted image.
% Output: img_ormp, img_lasso

% [N0, M0] = size(img_corr); 
m = para.m; gamma = para.gamma; % by default, should be 2/blsize2
blsize1 = para.blsize1; overlap1 = blsize1/2;
blsize2 = para.blsize2; overlap2 = blsize2/2;

% rem1 = mod(N0, blsize1/2); rem2 = mod(M0, blsize1/2); 
% if (rem1 ~= 0) || (rem2 ~= 0)
%     img_corr(:,M0+1:M0+rem2) = 0; corr(:,M0+1:M0+rem2) = 0;
%     img_corr(N0+1:N0+rem1,:) = 0; corr(N0+1:N0+rem1,:) = 0;
% end

[N,M] = size(img_corr);

% Dual Patch extraction
n_row = (N-overlap1)/(blsize1-overlap1); % number of patches per row
n_col = (M-overlap1)/(blsize1-overlap1);
n_patch = n_row*n_col;
count = 0;
for j = 0:n_row-1
    for k = 0:n_col-1
        count = count+1;
        img_patch(:,:,count) = img_corr([1:blsize1]+j*(blsize1-overlap1), [1:blsize1]+k*(blsize1-overlap1));
        corr_patch(:,:,count) = corr([1:blsize1]+j*(blsize1-overlap1), [1:blsize1]+k*(blsize1-overlap1));
    end
end

DCT=zeros(blsize2,sqrt(m));
for l=0:1:sqrt(m)-1
    V=cos([0:1:blsize2-1]'*l*pi/sqrt(m));
    if l>0 
        V=V-mean(V); 
    end
    DCT(:,l+1)=V/norm(V);
end 
D0=kron(DCT,DCT); 
% Use the same DCT dictionary for the initial dictionary.

iteration = 0; time = 0;
for k = 1:n_patch
    Y = patch_extract(img_patch(:,:,k), blsize2, overlap2); p = size(Y, 2);
    Corr = patch_extract(corr_patch(:,:,k), blsize2, overlap2);
    
    X0 = randn(m,p);
    opts.tol = 1e-4; opts.maxit = 10000;
    opts.D0 = D0; opts.X0 = X0;        
   
    % Dictionary learning, initialize with the DCT dictionary.
    % 0.5 seems to be better than 1
    fprintf('Patch%3i,',k);
    t0 = tic;
    [Dl, iter] = dl_apg(Y, Corr, m, gamma, opts);
    time = time+toc(t0); iteration = iteration + iter;

    % Sparse Coding with ORMP/ Least Square.
    X_ormp = zeros(m, p); X_lasso = zeros(m,p);
    op = struct('targetNonzeros', blsize2, 'verbose', 1);
    for l = 1:p
        X_ormp(:,l) = sparseapprox(Y(:,l), diag(Corr(:,l))*Dl, 'ORMP', op);
        X_lasso(:,l) = lasso(diag(Corr(:,l))*Dl, Y(:,l), 'lambda', 0.5/(m^2));
    end
    recovered_ormp = Dl*X_ormp; recovered_lasso = Dl*X_lasso;
    imgpatch_ormp(:,:,k) = patch2img_copy(recovered_ormp, blsize2, overlap2, [blsize1,blsize1]);
%     imgpatch_ormp0(:,:,k) = imgpatch_ormp(1:blsize1,1:blsize1,k);
    imgpatch_lasso(:,:,k) = patch2img_copy(recovered_lasso, blsize2, overlap2, [blsize1,blsize1]);
%     imgpatch_lasso0(:,:,k) = imgpatch_lasso(1:blsize1,1:blsize1,k);
end

%% Recover the image.
S = zeros(N,M);
count = 0;
img_ormp = zeros(N,M); img_lasso = zeros(N,M);
for k = 0:n_row-1
    for j = 0:n_col-1
        count = count+1;
        row_patch = [1:blsize1]+k*(blsize1-overlap1);
        col_patch = [1:blsize1]+j*(blsize1-overlap1);
        img_ormp(row_patch, col_patch)=img_ormp(row_patch, col_patch)+imgpatch_ormp(:,:,count);
        img_lasso(row_patch, col_patch)=img_lasso(row_patch, col_patch)+imgpatch_lasso(:,:,count);
        S(row_patch, col_patch) = S(row_patch, col_patch) + ones(blsize1, blsize1);
    end
end
img_ormp = img_ormp./S; 
img_lasso = img_lasso./S; 
end