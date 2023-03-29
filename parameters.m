%
% parameters

Nx     = 128;             % number of emitter elements
Ny     = 128;
ratio  = 2;               % ratio for rescaling input
ix     = round(Nx/ratio); % number of scaled emitter elements
iy     = round(Ny/ratio);
nx     = 1660e-6;         % size of array
ny     = 1660e-6;
d1     = 15e-2;           % propagation distance
d2     = 15e-2;
wv     = 663e-9;          % emission wavelength
a0     = 20;
r1     = nx/4;            % detector radius
r2     = nx/20;           % detector detection radius
rate   = 1;               % kernel update rate
lvalue = 1e-6;            % NaN prevention #
sx     = 2;               % tanh values
sy     = 1;
sc     = 0.45;
sz     = 0.2;
P      = 1;               % emission power in W
probability = 0.25;       % dropout probability

% training & testing images

filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

% get data for training

dataTrain = create_imagedatastore(filenameLabelsTrain, "TrainImagesPNG/");
dataTest  = create_imagedatastore(filenameLabelsTest , "TestImagesPNG/");

%dataTrain = create_xor_imagedatastore("XORLabels/train_labels.mat", "TrainImagesXOR");
%dataTest  = create_xor_imagedatastore("XORLabels/test_labels.mat" , "TestImagesXOR");

numEpochs = 32;           % number of epochs to iterate
miniBatchSize = 512;      % batch size per iteration
learnRate = 3e-4;         % learning rate

dataTest  = partition(dataTest , 10 , 1);     % test only first partition of test data
Sx = 28;                  % size of input images
Sy = 28;
C  = 1;                   % # of channels per input image

dimx = Sx;
dimy = Sy;

% get the interpolation value k
kx = log2(double(ix - dimy)/double(dimy - 1))+1;
ky = log2(double(iy - dimx)/double(dimx - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

w1 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d1, wv));
w2 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d2, wv));

kernel = internal_random_amp(Nx, Ny);
