clear; clc; close all;

rootFolder='D:/DeepLearning'; %%OWN PATH

classNames=["background","wound"];
labelIDs=[1 2];
inputSize=[224 224 3];
woundClassIdx=2;

numFolds=10;
maxImagesToProcess=10;

modelsRoot=fullfile(rootFolder,'MODELS','SEGNET_VGG19_RGB');

mainGradcamDir=fullfile(rootFolder,'GradCAM','SEGNET_VGG19_RGB');
if ~isfolder(mainGradcamDir)
    mkdir(mainGradcamDir);
end
fprintf('Main results folder: %s\n',mainGradcamDir);

for foldNum=1:numFolds

    fprintf('\n=============================================\n');
    fprintf('=== SEGNET VGG19 – FOLD %d / %d ===\n',foldNum,numFolds);
    fprintf('=============================================\n');

    foldModelName=fullfile(modelsRoot,sprintf('segnet_vgg19_validation_fold_%d.mat',foldNum));
    if ~isfile(foldModelName)
        warning('Model file not found: %s. Skipping Fold %d.\n',foldModelName,foldNum);
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
        warning('No test images in %s - skipping Fold %d.\n',testFoldPath,foldNum);
        continue;
    end

    thisMax=min(maxImagesToProcess,numTestImages);

    gradcamDir=fullfile(mainGradcamDir,sprintf('Fold%d',foldNum));
    if ~isfolder(gradcamDir)
        mkdir(gradcamDir);
    end
    fprintf('Results for Fold %d will be saved in: %s\n',foldNum,gradcamDir);

    for i=1:thisMax
        fprintf('\n=== Fold %d — Image %d / %d ===\n',foldNum,i,thisMax);

        I=readimage(imdsTestResized,i);
        trueMask=readimage(pxdsTestResized,i);

        predictedMask=semanticseg(I,net,'MiniBatchSize',1);

        [cam,featureLayerName,scoreLayerName]=gradCAM_segnet(net,I,woundClassIdx);

        fprintf('SegNet: featureLayer = %s, scoreLayer = %s\n',featureLayerName,scoreLayerName);

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

        hFig=figure('Visible','off','Position',[100 100 1800 400]);
        tiledlayout(1,4,'Padding','compact','TileSpacing','compact');

        nexttile;
        imshow(I);
        title('Test image','FontWeight','bold');

        nexttile;
        imshow(predOverlay);
        title('Predicted mask','FontWeight','bold');

        nexttile;
        imshow(overlayGlobalCAM);
        title('Global CAM (scoreLayer)','FontWeight','bold');

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

    fprintf('\nFinished Fold %d.\n',foldNum);
end

fprintf('\nFinished generating Grad-CAM (SegNet VGG19) for all folds.\n');

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

    tempImageDir=fullfile(tempdir,'ProcessedImages_SegNet');
    tempLabelDir=fullfile(tempdir,'ProcessedMasks_SegNet');
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
