%parameters_3

% parameters-2
%
% parameters

Nx     = 512;
Ny     = 512;
ratio  = 1;
ix     = round(Nx/ratio);
iy     = round(Ny/ratio);
nx     = 10.58e-3;
ny     = 10.58e-3;
d1     = 10e-2;
d2     = 10e-2;
wv     = 400e-9;
a0     = 20;
r1     = nx/6;
r2     = nx/30;
rate   = 1;
lvalue = 1e-6;
sx     = 2;
sy     = 1;
sc     = 0.45;
sz     = 2e-3;
P      = 1;
probability = 0.25;

filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

dataTrain = create_imagedatastore(filenameLabelsTrain, "DLPTrainImagesPNG/");
dataTest  = create_imagedatastore(filenameLabelsTest , "DLPTestImagesPNG/");

%dataTrain = create_xor_imagedatastore("XORLabels/train_labels.mat", "TrainImagesXOR");
%dataTest  = create_xor_imagedatastore("XORLabels/test_labels.mat" , "TestImagesXOR");

numEpochs = 64;
miniBatchSize = 128;
learnRate = (5e-4);

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

kernel = internal_random_amp(Nx, Ny);