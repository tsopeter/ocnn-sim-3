% CNN

clear;
parameters;

Rx = Nx/ratio;
Ry = Ny/ratio;
rx = nx/ratio;
ry = ny/ratio;

w1 = fftshift(get_propagation_distance(Rx, Ry, rx, ry, d1, wv));

InputLayer     = imageInputLayer([dimx, dimy, 1], 'Name', 'input', 'Normalization', 'none');
ResizeLayer    = CustomResizeLayer('resize', Nx, Ny, k, lvalue, P);
Prevention1    = CustomNaNPreventionLayer('prevention1', lvalue);
Prevention3    = CustomNaNPreventionLayer('prevention3', lvalue);
Prevention4    = CustomNaNPreventionLayer('prevention4', lvalue);
KernelLayer    = CustomKernelLayer('kernel', kernel, rate);
Polarizer      = CustomPolarizationLayer('polarizer');
Prop1          = convolution2dLayer([Rx Ry], 1, "Name", 'prop1', "Bias", lvalue, "Weights", real(w1), "Padding", "same", "PaddingValue", lvalue, "WeightLearnRateFactor", 0, "BiasLearnRateFactor", 0);
Prop2          = convolution2dLayer([Rx Ry], 1, "Name", 'prop2', "Bias", lvalue, "Weights", imag(w1), "Padding", "same", "PaddingValue", lvalue, "WeightLearnRateFactor", 0, "BiasLearnRateFactor", 0);
Non1           = CustomNonlinearLayer('non1', lvalue, sx, sy, sc, sz);
Prevention2    = CustomNaNPreventionLayer('prevention2', lvalue);
Abs1           = CustomAbsoluteLayer(2, 'abs1', 0);
Flatten        = CustomFlattenLayer(1, 'flatten', Nx, Ny, nx, ny, r1, r2, 0);
Softmax        = softmaxLayer("Name", 'softmax');
Classification = classificationLayer("Name", 'classification');

layers = [
    InputLayer
    ResizeLayer
    Prevention1
    KernelLayer
    Prevention3
    Prevention4
    Polarizer
    Prop1
    Prop2
    Abs1
    Non1
    Flatten
    Softmax
    Classification
];

lgraph = layerGraph();
for i=1:length(layers)
    lgraph = addLayers(lgraph, layers(i));
end

lgraph = connectLayers(lgraph, 'input', 'resize');
lgraph = connectLayers(lgraph, 'resize', 'prevention1');
lgraph = connectLayers(lgraph, 'prevention1', 'kernel');
lgraph = connectLayers(lgraph, 'kernel/out1', 'prevention3');
lgraph = connectLayers(lgraph, 'kernel/out2', 'prevention4');
lgraph = connectLayers(lgraph, 'prevention3', 'polarizer/in1');
lgraph = connectLayers(lgraph, 'prevention4', 'polarizer/in2');
lgraph = connectLayers(lgraph, 'polarizer', 'prop1');
lgraph = connectLayers(lgraph, 'polarizer', 'prop2');
lgraph = connectLayers(lgraph, 'prop1', 'abs1/in1');
lgraph = connectLayers(lgraph, 'prop2', 'abs1/in2');
lgraph = connectLayers(lgraph, 'abs1', 'non1');
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
