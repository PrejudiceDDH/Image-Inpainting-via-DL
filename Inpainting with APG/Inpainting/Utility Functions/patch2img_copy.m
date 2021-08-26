function I = patch2img_copy(X, blsize, overlap, im_size)
% Recover overlapping (blsize x blsize) patches into an image reconstruction
% Only recover the corrupted part. 

% X: (blsize^2 x p) matrix containing the vectorised signals
% blsize: patch size
% overlap: number of overlapping pixels of two consecutive patches
% im_size: tuple of original image dimensions

% Get image dimensions
N = im_size(1);
M = im_size(2);

% Initialise the image
I = zeros(N, M);

% Initialise the matrix that stores the cumulative signal used to calculate
% each pixel
S = zeros(N, M);

% Calculate how many patches fit horisontally and vertically
row = (N - overlap)/(blsize-overlap);
col = (M - overlap)/(blsize-overlap);

% Iterate through patches and keep a counter to know what column of X to
% access next
counter = 0;
for i = 0:row-1
    for j = 0:col-1
        counter = counter + 1; % record the # pathches
        % Get x and y range of current patch
        x_patch = [1:blsize]+i*(blsize-overlap);
        y_patch = [1:blsize]+j*(blsize-overlap);
        % Get current patch and calculate its signal strength
        current_patch = reshape(X(:,counter), blsize, blsize);
        % Update cumulative number of covers
        S(x_patch, y_patch) = S(x_patch, y_patch) + ones(blsize, blsize);
        % Update patch
        I(x_patch, y_patch) = I(x_patch, y_patch) + current_patch;
    end
end
% If a pixel was not covered by a single patch with signal, set its signal
% value to 1, in order to prevent division by zero
S(S == 0) = 1;

% Divide each pixel by its corresponding cumulative signal coverage
I = I./S;
end
