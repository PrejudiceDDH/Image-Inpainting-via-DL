function Y = patch_extract(img, blsize, overlap)
% Extracts (bb x bb) patches from image I after every 'overlap' pixels
% blsize: patch size
% overlap: number of overlapping pixels of two consecutive patches

% Get image size
[N, M] = size(img);
% Initialise Y
Y = [];
% Calculate how many patches fit horisontally and vertically
row = (N - overlap)/(blsize-overlap);
col = (M - overlap)/(blsize-overlap);

% Iterate through patches
for i = 0:row-1
    for j = 0:col-1
        % Take a patch with top-left corner at (i, j) and vectorise it
        % fprintf('%bb  %bb | %bb  %bb \n',1+i*overlap,bb+i*overlap,1+j*overlap,bb+j*overlap)
        Y = [Y, reshape(img([1:blsize]+i*(blsize-overlap), [1:blsize]+j*(blsize-overlap)), blsize*blsize, 1)];
    end
end
end

