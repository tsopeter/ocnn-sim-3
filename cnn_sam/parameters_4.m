%parameters 4

% Parameters 4 is based on the following paramter format
% - parameters_3
% - parameters-2
% - parameters

%
% note that the base measurement are
% mW (milliWatts)
% M (Meters)

Nx     = 512;               % size of whatever screen is used
Ny     = 512;
ratio  = 1;                 % ratio factor
ix     = round(Nx/ratio);
iy     = round(Ny/ratio);

nx     = 21.16e-3;          % 21.16 mm, assumed size of DLP
ny     = 21.16e-3;
d1     = 1.4e-3;            % distance of 1.4 mm, this is the focal
d2     = 1.4e-3;            % length of the SAM (Saturable Absorbption Mirror)
wv     = 1550e-9;           % use 1550 nm wavelength
a0     = 20;                % NOT USED

r1     = nx/6;              % specifies detection zone and size  
r2     = nx/30;

rate   = 1;
lvalue = 1e-6;

sx     = 2;                 % not used (used or modified tanh function)
sy     = 1;
sc     = 0.45;
sz     = 1/6;

P      = 1;                 % default power of 1 mW
mm     = 2 * 10e3;          % For use in DLPEndLayer. Larger the factor, means sharper response
probability = 0.25;         % For use in DLPLayer. This factor allows dithering in 1-bit DLPs
IPower = 8;                 % start with 8 mW

filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

dataTrain = create_imagedatastore(filenameLabelsTrain, "DLPTrainImagesPNG/");
dataTest  = create_imagedatastore(filenameLabelsTest , "DLPTestImagesPNG/");

%dataTrain = create_xor_imagedatastore("XORLabels/train_labels.mat", "TrainImagesXOR");
%dataTest  = create_xor_imagedatastore("XORLabels/test_labels.mat" , "TestImagesXOR");

numEpochs = 64;             % number of epochs
miniBatchSize = 128;        % size of batches per epoch
learnRate = (1e-4);         % learn rate (reduce when NaNs occur)

dataTest  = partition(dataTest , 10 , 1);
Sx = 256;
Sy = 256;
C  = 1;

dimx = Sx;
dimy = Sy;

% get the interpolation value k
kx = log2(double(ix - dimy)/double(dimy - 1))+1;
ky = log2(double(iy - dimx)/double(dimx - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

w1 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d1, wv));
w2 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d2, wv));

kernel = internal_random_amp(Nx, Ny)*IPower;        % create the kernel