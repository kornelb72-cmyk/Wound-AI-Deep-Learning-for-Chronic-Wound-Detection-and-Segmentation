clear; clc; close all;

rootFolder ='C:/Users/korne/Desktop/DEMO';  
imgName ='case_6_1_1_rgb.png';
maskName ='case_6_1_1_gt.png';

classNames = ["background","wound"];
labelIDs = [1 2];              
inputSize = [224 224 3];

maxEpochs = 50;
miniBatchSize = 1;             
initialLearnRate = 1e-4;

imgPath = fullfile(rootFolder, imgName);
maskPath = fullfile(rootFolder, maskName);

imds = imageDatastore({imgPath}, ...
    'ReadFcn', @(f)readAndPreprocessImage(f, inputSize));

pxds = pixelLabelDatastore({maskPath}, classNames, labelIDs, ...
    'ReadFcn', @(f)readAndPreprocessLabel(f, inputSize, labelIDs));

trainingData = pixelLabelImageDatastore(imds, pxds);
validationData = trainingData;  

lgraph = deeplabv3plusLayers(inputSize,numel(classNames),'resnet50');

pxLayer = pixelClassificationLayer('Classes',classNames,'Name','pixelOutput');
lgraph = replaceLayer(lgraph, 'classification', pxLayer);

analyzeNetwork(lgraph);

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'L2Regularization', 1e-4, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 1, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 1);

fprintf('Training (1 image) ...\n');
[net, info] = trainNetwork(trainingData, lgraph, options);

modelFile = fullfile(rootFolder,'deeplabv3plus_resnet50_case_6_1_1_demo.mat');
save(modelFile,'net','info');
fprintf('Model was saved in: %s\n', modelFile);

function I = readAndPreprocessImage(filename, inputSize)
    I = imread(filename);
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    I = imresize(I, inputSize(1:2));
end

function L = readAndPreprocessLabel(filename, inputSize, labelIDs)

    M = imread(filename);

    if size(M,3) > 1
        M = M(:,:,1);
    end

    M = imresize(M, inputSize(1:2), 'nearest');
    M = uint8(M);

    u = unique(M(:));

    if isequal(u, uint8([0;1])) || isequal(u, uint8([0 1]))
        M = M + 1;
    elseif isequal(u, uint8([0;255])) || isequal(u, uint8([0 255]))
        M(M==0)   = 1;
        M(M==255) = 2;
    else
        if numel(u) >= 2
            M(M==min(u)) = 1;
            M(M==max(u)) = 2;
        end
    end

    M(~ismember(M, uint8(labelIDs))) = uint8(labelIDs(1));
    L = M;
end