clear; clc; close all;

rootFolder='D:/DeepLearning';   %%CHANGE ON OWN PATH
numFolds=10;
classNames=["background","wound"];
labelIDs=[1,2];
inputSize=[224,224,3];
maxEpochs=50;
initialLearnRate=1e-3;

modelsRoot=fullfile(rootFolder,'MODELS','DEEPLAB_ResNet50_RGB_SGDM');
if ~isfolder(modelsRoot)
    mkdir(modelsRoot);
end

fprintf('Models will be saved in: %s\n',modelsRoot);

for foldNum=1:numFolds
    fprintf('\nProcessing Fold %d...\n',foldNum);

    trainingFoldNames=arrayfun(@(x)sprintf('Fold%d',x),setdiff(1:numFolds,foldNum),'UniformOutput',false);

    [imdsTrainResized,pxdsTrainResized]=prepareFoldData(trainingFoldNames,inputSize,classNames,labelIDs,rootFolder);
    [imdsTestResized,pxdsTestResized]=prepareFoldData({sprintf('Fold%d',foldNum)},inputSize,classNames,labelIDs,rootFolder);

    numFiles=numel(imdsTrainResized.Files);
    idx=randperm(numFiles);
    splitPoint=round(0.8*numFiles);

    trainIdx=idx(1:splitPoint);
    valIdx=idx(splitPoint+1:end);

    imdsTrainSplit=subset(imdsTrainResized,trainIdx);
    pxdsTrainSplit=subset(pxdsTrainResized,trainIdx);
    imdsValidationSplit=subset(imdsTrainResized,valIdx);
    pxdsValidationSplit=subset(pxdsTrainResized,valIdx);

    trainingData=pixelLabelImageDatastore(imdsTrainSplit,pxdsTrainSplit);
    validationData=pixelLabelImageDatastore(imdsValidationSplit,pxdsValidationSplit);

    lgraph=deeplabv3plusLayers(inputSize,numel(classNames),'resnet50');

    pxLayer=pixelClassificationLayer('Classes',classNames,'Name','pixelOutput');
    lgraph=replaceLayer(lgraph,'classification',pxLayer);

    analyzeNetwork(lgraph);

    options=trainingOptions('sgdm', ...
        'InitialLearnRate',initialLearnRate, ...
        'L2Regularization',1e-3, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',8, ...
        'Shuffle','every-epoch', ...
        'ValidationData',validationData, ...
        'ValidationFrequency',10, ...
        'Plots','training-progress', ...
        'VerboseFrequency',10);

    try
        fprintf('Starting training for Fold %d...\n',foldNum);
        [net,info]=trainNetwork(trainingData,lgraph,options);

        foldModelName=fullfile(modelsRoot,sprintf('deeplabv3plus_resnet50_validation_sgdm_fold_%d.mat',foldNum));
        save(foldModelName,'net','info');
        fprintf('Model for Fold %d has been saved as %s\n',foldNum,foldModelName);
    catch ME
        fprintf('Training was interrupted in Fold %d!\n',foldNum);
        if exist('net','var')
            foldPartialModelName=fullfile(modelsRoot,sprintf('deeplabv3plus_resnet50_validation_sgdm_fold_%d_partial.mat',foldNum));
            save(foldPartialModelName,'net','info');
            fprintf('Partially trained model for Fold %d has been saved as %s\n',foldNum,foldPartialModelName);
        end
        rethrow(ME);
    end
end

function [imdsResized,pxdsResized]=prepareFoldData(foldNames,inputSize,classNames,labelIDs,rootFolder)
    images={};
    labels={};
    for i=1:numel(foldNames)
        foldPath=fullfile(rootFolder,foldNames{i});
        caseDirs=dir(fullfile(foldPath,'case_*'));
        for j=1:numel(caseDirs)
            casePath=fullfile(foldPath,caseDirs(j).name);
            imageFiles=dir(fullfile(casePath,'*_rgb.png'));
            labelFiles=dir(fullfile(casePath,'*_gt.png'));
            if ~isempty(imageFiles) && ~isempty(labelFiles)
                imagePaths=fullfile({imageFiles.folder},{imageFiles.name});
                labelPaths=fullfile({labelFiles.folder},{labelFiles.name});
                images=[images;imagePaths(:)];
                labels=[labels;labelPaths(:)];
            end
        end
    end

    [imagesSynced,labelsSynced]=synchronizeDatasets(images,labels,'_rgb','_gt');

    resizedImages=cellfun(@(x)ensureRGB(imresize(imread(x),inputSize(1:2))), ...
                          imagesSynced,'UniformOutput',false);
    resizedMasks=cellfun(@(x)uint8(imresize(imread(x),inputSize(1:2),'nearest')), ...
                         labelsSynced,'UniformOutput',false);

    tempImageDir=tempname;
    tempLabelDir=tempname;
    mkdir(tempImageDir);
    mkdir(tempLabelDir);
    for i=1:numel(resizedImages)
        imwrite(resizedImages{i},fullfile(tempImageDir,sprintf('image_%d.png',i)));
        imwrite(resizedMasks{i},fullfile(tempLabelDir,sprintf('label_%d.png',i)));
    end

    imdsResized=imageDatastore(tempImageDir);
    pxdsResized=pixelLabelDatastore(tempLabelDir,classNames,labelIDs);
end

function [syncedImages,syncedLabels]=synchronizeDatasets(images,labels,suffixImage,suffixLabel)
    [~,imageNames,~]=cellfun(@fileparts,images,'UniformOutput',false);
    [~,labelNames,~]=cellfun(@fileparts,labels,'UniformOutput',false);
    imageNames=strrep(imageNames,suffixImage,'');
    labelNames=strrep(labelNames,suffixLabel,'');
    [~,imgIdx,lblIdx]=intersect(imageNames,labelNames);
    syncedImages=images(imgIdx);
    syncedLabels=labels(lblIdx);
end

function imgRGB=ensureRGB(img)
    if size(img,3)==1
        imgRGB=repmat(img,[1 1 3]);
    else
        imgRGB=img;
    end
end
