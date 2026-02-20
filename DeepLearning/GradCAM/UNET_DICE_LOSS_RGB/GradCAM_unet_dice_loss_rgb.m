clear; clc; close all;

rootFolder='D:\DeepLearning';  %% OWN PATH

classNames=["background","wound"];
labelIDs=[1 2];

inputSize=[128 128 3];
woundClassIdx=2;

numFolds=10;
maxImagesToProcess=10;

modelsRoot=fullfile(rootFolder,'MODELS','UNET_DICE_LOSS_RGB');

mainGradcamDir=fullfile(rootFolder,'GradCAM','UNET_DICE_LOSS_RGB');
if ~isfolder(mainGradcamDir)
    mkdir(mainGradcamDir);
end
fprintf('Main results folder: %s\n',mainGradcamDir);

for foldNum=1:numFolds

    fprintf('\n=============================\n');
    fprintf('=== FOLD %d / %d ===\n',foldNum,numFolds);
    fprintf('=============================\n');

    foldModelName=fullfile(modelsRoot,sprintf('unet_dice_loss_fold_%d.mat',foldNum));
    if ~isfile(foldModelName)
        warning('Model file not found: %s. Skipping this fold.\n',foldModelName);
        continue;
    end

    load(foldModelName,'net');
    fprintf('Loaded model: %s\n',foldModelName);

    testFoldName=sprintf('Fold%d',foldNum);
    testFoldPath=fullfile(rootFolder,testFoldName);

    [imdsTestResized,pxdsTestResized]=loadAndProcessTestData(testFoldPath,classNames,labelIDs,inputSize);

    numTestImages=numel(imdsTestResized.Files);
    fprintf('Found %d test images.\n',numTestImages);

    if numTestImages==0
        warning('No test images in %s. Skipping this fold.\n',testFoldName);
        continue;
    end

    thisMaxImagesToProcess=min(maxImagesToProcess,numTestImages);

    gradcamDir=fullfile(mainGradcamDir,sprintf('Fold%d',foldNum));
    if ~isfolder(gradcamDir)
        mkdir(gradcamDir);
    end
    fprintf('Results for Fold %d will be saved in: %s\n',foldNum,gradcamDir);

    for i=1:thisMaxImagesToProcess
        fprintf('\n=== Fold %d — Image %d / %d ===\n',foldNum,i,thisMaxImagesToProcess);

        I=readimage(imdsTestResized,i);
        trueMask=readimage(pxdsTestResized,i); 

        predictedMask=semanticseg(I,net,'MiniBatchSize',1);

        [cam,featureLayerName,scoreLayerName]=gradCAM_unet(net,I,woundClassIdx);

        fprintf('UNet (Fold %d): featureLayer = %s, scoreLayer = %s\n',foldNum,featureLayerName,scoreLayerName);

        woundMaskPred=predictedMask=='wound';

        scores=activations(net,I,scoreLayerName);
        woundMap=scores(:,:,woundClassIdx);

        woundMaskSmall=imresize(single(woundMaskPred), ...
                                [size(woundMap,1) size(woundMap,2)], ...
                                'nearest');

        woundMapMasked=woundMap.*woundMaskSmall;
        woundMapMasked=mat2gray(woundMapMasked);
        woundMapMaskedUp=imresize(woundMapMasked,[size(I,1) size(I,2)]);

        woundCAM_RGB=ind2rgb(gray2ind(woundMapMaskedUp,256),jet(256));
        overlayMaskedCAM=0.5*woundCAM_RGB+0.5*im2double(I);

        woundMapGlobal=mat2gray(woundMap);
        woundMapGlobalUp=imresize(woundMapGlobal,[size(I,1) size(I,2)]);

        woundGlobalCAM_RGB=ind2rgb(gray2ind(woundMapGlobalUp,256),jet(256));
        overlayGlobalCAM=0.5*woundGlobalCAM_RGB+0.5*im2double(I);

        camNorm=mat2gray(cam);
        camRGB=ind2rgb(gray2ind(camNorm,256),jet(256));
        overlayGradCAM=0.5*camRGB+0.5*im2double(I);

        predOverlay=labeloverlay(I,predictedMask,'Transparency',0.5);

        hFig=figure('Visible','off','Position',[100 100 2000 400]);
        tiledlayout(1,5,'Padding','compact','TileSpacing','compact');

        nexttile; 
        imshow(I);
        title('Test image','FontWeight','bold');
        nexttile; 
        imshow(predOverlay);     
        title('Predicted mask','FontWeight','bold');
        nexttile; 
        imshow(overlayGradCAM);  
        title('Gradient Grad-CAM','FontWeight','bold');
        nexttile; 
        imshow(overlayGlobalCAM);
        title('Global CAM','FontWeight','bold');
        nexttile; 
        imshow(overlayMaskedCAM);
        title('CAM within wound region','FontWeight','bold');

        baseName=sprintf('Fold%d_img_%03d',foldNum,i);

        imwrite(camNorm,fullfile(gradcamDir,[baseName '_cam_gray.png']));
        imwrite(camRGB,fullfile(gradcamDir,[baseName '_cam_color.png']));
        imwrite(overlayGradCAM,fullfile(gradcamDir,[baseName '_overlay_gradcam.png']));
        imwrite(predOverlay,fullfile(gradcamDir,[baseName '_overlay_predMask.png']));
        imwrite(overlayGlobalCAM,fullfile(gradcamDir,[baseName '_overlay_globalCAM.png']));
        imwrite(overlayMaskedCAM,fullfile(gradcamDir,[baseName '_overlay_maskedCAM.png']));

        saveas(hFig,fullfile(gradcamDir,[baseName '_summary.png']));
        close(hFig);

        save(fullfile(gradcamDir,[baseName '_cam.mat']),'cam');
    end

    fprintf('\nFinished Grad-CAM for Fold %d (U-Net).\n',foldNum);
end

fprintf('\nFinished generating Grad-CAM for all folds (U-Net).\n');

function [imdsResized,pxdsResized]=loadAndProcessTestData(testFoldPath,classNames,labelIDs,inputSize)
    fprintf('Loading and processing test data from: %s\n',testFoldPath);

    imageFiles=dir(fullfile(testFoldPath,'**','*_rgb.png'));
    labelFiles=dir(fullfile(testFoldPath,'**','*_gt.png'));

    if isempty(imageFiles) || isempty(labelFiles)
        error('No images or masks in folder: %s',testFoldPath);
    end

    imagePaths=fullfile({imageFiles.folder},{imageFiles.name});
    labelPaths=fullfile({labelFiles.folder},{labelFiles.name});

    [imagePathsSynced,labelPathsSynced]=synchronizeDatasets(imagePaths,labelPaths,'_rgb','_gt');

    if isempty(imagePathsSynced) || isempty(labelPathsSynced)
        error('No matching images and masks in folder: %s',testFoldPath);
    end

    tempImageDir=fullfile(tempdir,'ProcessedImages_UNet');
    tempLabelDir=fullfile(tempdir,'ProcessedMasks_UNet');
    if ~isfolder(tempImageDir), mkdir(tempImageDir); end
    if ~isfolder(tempLabelDir), mkdir(tempLabelDir); end

    for i=1:numel(imagePathsSynced)
        img=imread(imagePathsSynced{i});
        imgResized=imresize(img,inputSize(1:2));
        if size(imgResized,3)==1
            imgResized=repmat(imgResized,[1 1 3]);
        end
        imwrite(imgResized,fullfile(tempImageDir,sprintf('image_%d.png',i)));

        mask=imread(labelPathsSynced{i});
        if size(mask,3)>1
            mask=mask(:,:,1);
        end
        maskResized=imresize(mask,inputSize(1:2),'nearest');
        imwrite(uint8(maskResized),fullfile(tempLabelDir,sprintf('mask_%d.png',i)));
    end

    imdsResized=imageDatastore(tempImageDir);
    pxdsResized=pixelLabelDatastore(tempLabelDir,classNames,labelIDs);

    validateDataset(imdsResized,pxdsResized,inputSize);
end

function validateDataset(imds,pxds,inputSize)
    fprintf('Validating image and mask sizes...\n');

    for i=1:numel(imds.Files)
        img=imread(imds.Files{i});
        if ~isequal(size(img,1:2),inputSize(1:2))
            error('Image %s has size [%d, %d], expected [%d, %d].',imds.Files{i},size(img,1),size(img,2),inputSize(1),inputSize(2));
        end
    end

    for i=1:numel(pxds.Files)
        mask=imread(pxds.Files{i});
        if ~isequal(size(mask,1:2),inputSize(1:2))
            error('Mask %s has size [%d, %d], expected [%d, %d].',pxds.Files{i},size(mask,1),size(mask,2),inputSize(1),inputSize(2));
        end
        if size(mask,3)~=1
            error('Mask %s has an invalid number of channels: %d. Expected 1.',pxds.Files{i},size(mask,3));
        end
    end
end

function [imagesSynced,labelsSynced]=synchronizeDatasets(images,labels,suffixImage,suffixLabel)
    [~,imageNames,~]=cellfun(@fileparts,images,'UniformOutput',false);
    [~,labelNames,~]=cellfun(@fileparts,labels,'UniformOutput',false);
    imageNamesClean=strrep(imageNames,suffixImage,'');
    labelNamesClean=strrep(labelNames,suffixLabel,'');
    [commonNames,imgIdx,lblIdx]=intersect(imageNamesClean,labelNamesClean,'stable');
    imagesSynced=images(imgIdx);
    labelsSynced=labels(lblIdx);

    if isempty(commonNames)
        warning('No matching pairs of images and masks were found.');
    end
end
