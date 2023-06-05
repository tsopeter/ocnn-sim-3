% cnn-sam-non-ideal
% based on cnn-4

clear;
clc;
parameters_4;

%
% If the folders
% DLPTrainImagesPNG
% DLPTestImagesPNG
% are not yet created and filled
% , create folders
% and run fast_create_images_script to
% populate it

addpath(genpath('SAM data'));
[P0, D0] = SAM_polyn_mW();
xnormal = 1;
ynormal = 1;

%
% NOTE
% the base unit of measurement is mW
% not W like in cnn, cnn_2, cnn_3, or cnn_4
% this is to prevent NaN's from happening during
% training and validation.
%
% therefore, use mW versions of the same
% functions, i.e., use SAM_polyn_mW, and
% CustomFFT2mWPropagationLayer
%
% Other layers such as DLPEndLayer
% and others do not depend on it
%
% Note: With DLPEndLayer
% this caps the maximum Amplitude to 1 mW and
% the minimum ampitude to 0 mW.

Nx = dimx;  % reset resolution from 512 to 256, effectively speeding up training by 4x
Ny = dimy;

% recompute angluar spectrum sources and kernel
w1 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d1, wv));
w2 = fftshift(get_propagation_distance(Nx, Ny, nx, ny, d2, wv));
kernel = internal_random_amp(Nx, Ny)*IPower;        % create the kernel

InputLayer     = imageInputLayer([dimx, dimy, 1], 'Name', 'input', 'Normalization', 'rescale-zero-one');
ResizeLayer    = resize2dLayer("Name", 'resize', 'OutputSize', [Nx Ny], 'Method','nearest', 'NearestRoundingMode','round');
%KernelLayer    = CustomAmplitudeKernelLayer('kernel', randn([Nx, Ny]));
%DLPEndLayer    = CustomDLPEndLayer('dlp_end', Nx, Ny, mm);

Prevention0A    = CustomNaNPreventionLayer('prevention0A', lvalue);
Prevention0B    = CustomNaNPreventionLayer('prevention0B', lvalue);

KernelLayer    = CustomPhaseKernelLayer('kernel', kernel, lvalue);
Polarizer      = CustomPolarizationLayer('polarizer');

Prevention1    = CustomNaNPreventionLayer('prevention1', lvalue);
Prop1          = CustomFFT2mWPropagationLayer('prop1', Nx, Ny, nx, ny, d1, wv);
%Non1           = CustomNonlinearLayer('non1', lvalue, sx, sy, sc, sz);
DUT            = CustomPolynomialNonLinearLayer('DUT', P0, D0, [Nx, Ny, 1], xnormal, ynormal);
Flatten        = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);
Softmax        = softmaxLayer("Name", 'softmax');
Classification = classificationLayer("Name", 'classification');
layers = [
    InputLayer
    %ResizeLayer % remove resize layer and train on 256x256 (more than
                 %enough resolution
    KernelLayer
    Prevention0A
    Prevention0B
    %DLPEndLayer
    Polarizer
    Prevention1
    Prop1
    %Non1
    DUT
    Flatten
    Softmax
    Classification
];

lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

%lgraph = connectLayers(lgraph, 'input', 'resize');
%lgraph = connectLayers(lgraph, 'resize', 'kernel');

lgraph = connectLayers(lgraph, 'input', 'kernel');

lgraph = connectLayers(lgraph, 'kernel/out1', 'prevention0A');
lgraph = connectLayers(lgraph, 'kernel/out2', 'prevention0B');
lgraph = connectLayers(lgraph, 'prevention0A', 'polarizer/in1');
lgraph = connectLayers(lgraph, 'prevention0B', 'polarizer/in2');
lgraph = connectLayers(lgraph, 'polarizer', 'prevention1');
lgraph = connectLayers(lgraph, 'prevention1', 'prop1');
lgraph = connectLayers(lgraph, 'prop1', 'DUT');
lgraph = connectLayers(lgraph, 'DUT', 'flatten');
lgraph = connectLayers(lgraph, 'flatten', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classification');

plot(lgraph);

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    MaxEpochs=numEpochs,...
    Shuffle='every-epoch',...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='auto',...
    DispatchInBackground=false,...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain,lgraph,options);

YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)