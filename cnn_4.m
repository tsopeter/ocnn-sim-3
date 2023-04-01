% cnn 4
% CNN

clear;
parameters_3;

InputLayer     = imageInputLayer([dimx, dimy, 1], 'Name', 'input', 'Normalization', 'rescale-zero-one');
KernelLayer    = CustomAmplitudeKernelLayer('kernel', real(kernel));
DLPEndLayer    = CustomDLPEndLayer('dlp_end', Nx, Ny);
Prevention1    = CustomNaNPreventionLayer('prevention1', lvalue);
Prop1          = CustomPropagationLayer('prop1', Nx, Ny, nx, ny, d1, wv);
Non1           = CustomNonlinearLayer('non1', lvalue, sx, sy, sc, sz);
Flatten        = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, lvalue);
Softmax        = softmaxLayer("Name", 'softmax');
Classification = classificationLayer("Name", 'classification');
layers = [
    InputLayer
    KernelLayer
    DLPEndLayer
    Prevention1
    Prop1
    Non1
    Flatten
    Softmax
    Classification
];

lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input', 'kernel');
lgraph = connectLayers(lgraph, 'kernel', 'dlp_end');
lgraph = connectLayers(lgraph, 'dlp_end', 'prevention1');
lgraph = connectLayers(lgraph, 'prevention1', 'prop1');
lgraph = connectLayers(lgraph, 'prop1', 'non1');
lgraph = connectLayers(lgraph, 'non1', 'flatten');
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
    DispatchInBackground=true,...
    MiniBatchSize=miniBatchSize);

net = trainNetwork(dataTrain,lgraph,options);

YPred = classify(net,dataTest);
YValidation = dataTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)