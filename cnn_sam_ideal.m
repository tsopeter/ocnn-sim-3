% cnn-sam-ideal

clear;
clc;

%
% this is a simple 2x2 network to test just SAM reflectivity

% parameters
ss     = [28, 28, 1];
P      = 1e-3;   % default power of 1 mW
PMax   = 20e-3;  % maximum power of 20 mW
lvalue = 1e-6;   % minimum power of 1 uW
RR     = 100;    % rise-rate value for emulating DLP

filenameImagesTrain = 'Images/train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'Images/train-labels-idx1-ubyte.gz';
filenameImagesTest  = 'Images/t10k-images-idx3-ubyte.gz';
filenameLabelsTest  = 'Images/t10k-labels-idx1-ubyte.gz';

dataTrain = create_imagedatastore(filenameLabelsTrain, "SAMTrainImagesPNG/");
dataTest  = create_imagedatastore(filenameLabelsTest , "SAMTestImagesPNG/");
dataTest  = partition(dataTest , 10 , 1);   % to reduce validation time

learnRate     = 5e-3;
maxEpochs     = 4;
numEpochs     = 4;
miniBatchSize = 128;

%
%   | A | B |
%   | C | D |
%

addpath(genpath('SAM data'));
[P0, D0] = SAM_poly();

kernel   = randn(ss);

% we take the input
InputLayer     = imageInputLayer(ss, 'Name', 'input', 'Normalization', 'rescale-zero-one');
Kernel         = CustomAmplitudeKernelLayer('Kernel', kernel);
DUT            = CustomPolynomialNonLinearLayer('DUT', P0, D0, ss(1));
DUT2           = CustomPolynomialNonLinearLayer('DUT', [1, 0], 0, ss(1));
Prevention     = CustomNaNPreventionLayer('prevention', lvalue);
Flatten        = flattenLayer('Name', 'flatten');
Linear         = fullyConnectedLayer(10, 'Name', 'linear', 'BiasInitializer', 'zeros', 'WeightsInitializer', 'zeros');
SoftMax        = softmaxLayer('Name', 'softmax');
Classification = classificationLayer('Name', 'classification');

layers = [
    InputLayer
    Kernel
    DUT
    %DUT2
    Prevention
    Flatten
    Linear
    SoftMax
    Classification
];
lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input', 'Kernel');
lgraph = connectLayers(lgraph, 'Kernel', 'DUT');
lgraph = connectLayers(lgraph, 'DUT', 'prevention');
lgraph = connectLayers(lgraph, 'prevention', 'flatten');
lgraph = connectLayers(lgraph, 'flatten', 'linear');
lgraph = connectLayers(lgraph, 'linear', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classification');

plot(lgraph);

%
%
% the structure should be
%
%   [Input]->[Kernel]->[SAM]->[Flatten]->[SoftMax]->Classification
%   
%   Note that power is not capped yet,
%   if there are any visible issues, then we should cap it

options = trainingOptions('adam', ...
    InitialLearnRate=learnRate,...
    MaxEpochs=maxEpochs, ...
    Shuffle='every-epoch',...
    ValidationData=dataTest,...
    ValidationFrequency=512,...
    Verbose=true,...
    Plots='training-progress',...
    ExecutionEnvironment='auto',...
    DispatchInBackground=false,...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain, lgraph, options);
YPred = classify(net, dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred==YValidation)/numel(YValidation)