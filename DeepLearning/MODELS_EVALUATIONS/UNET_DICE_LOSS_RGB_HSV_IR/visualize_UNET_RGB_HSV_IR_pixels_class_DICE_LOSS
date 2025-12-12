clear;clc;close all;

rootFolder='D:/DeepLearning';  %%OWN PATH

evalRoot=fullfile(rootFolder,'MODELS_EVALUATIONS','UNET_DICE_LOSS_RGB_HSV_IR');
if ~isfolder(evalRoot),mkdir(evalRoot);end

modelsRoot=fullfile(rootFolder,'MODELS','UNET_DICE_LOSS_RGB_HSV_IR');
if ~isfolder(modelsRoot)
error('Models folder does not exist: %s',modelsRoot);
end

segResultsRoot=fullfile(evalRoot,'Segmentation_Results_UNET_RGB_HSV_IR_DICE_LOSS');
if ~isfolder(segResultsRoot),mkdir(segResultsRoot);end

visRoot=fullfile(segResultsRoot,'Visualization_Results');
woundRoot=fullfile(segResultsRoot,'WoundMasks');
statsRoot=fullfile(segResultsRoot,'Statistics');
tissueRoot=fullfile(statsRoot,'TISSUE_CLASSIFICATION');

foldersToMake={visRoot,woundRoot,statsRoot,tissueRoot};
for k=1:numel(foldersToMake)
if ~isfolder(foldersToMake{k}),mkdir(foldersToMake{k});end
end

numFolds=10;
classNames=["background","wound"];
labelIDs=[1,2];
inputSize=[224,224,3];

for foldNum=1:numFolds
fprintf('\nEvaluating model for Fold %d...\n',foldNum);
foldModelName=fullfile(modelsRoot,sprintf('unet_dice_loss_rgbhsvir_fold_%d.mat',foldNum));
if isfile(foldModelName)
load(foldModelName,'net');
fprintf('Model for Fold %d has been loaded from %s.\n',foldNum,foldModelName);
fprintf('Preparing test data for Fold %d...\n',foldNum);
testFoldName=sprintf('Fold%d',foldNum);
testFoldPath=fullfile(rootFolder,testFoldName);

imageFiles=dir(fullfile(testFoldPath,'**','*_obraz.*'));
labelFiles=dir(fullfile(testFoldPath,'**','*_gt.*'));
testImages=fullfile({imageFiles.folder},{imageFiles.name})';
testLabels=fullfile({labelFiles.folder},{labelFiles.name})';


[testImagesSynced,testLabelsSynced]=synchronizeDatasets(testImages,testLabels,'_obraz','_gt');

origImagePaths=testImagesSynced(:);
imdsTest=imageDatastore(testImagesSynced);
pxdsTest=pixelLabelDatastore(testLabelsSynced,classNames,labelIDs);

fprintf('Resizing test data...\n');
resizedImagesTest=cellfun(@(x)ensureRGB(imresize(imread(x),inputSize(1:2))),imdsTest.Files,'UniformOutput',false);
resizedMasksTest=cellfun(@(x)normalizeGT(imresize(imread(x),inputSize(1:2),'nearest')),pxdsTest.Files,'UniformOutput',false);

tempImageDirTest=fullfile(tempdir,sprintf('ResizedImagesTest_Fold%d',foldNum));
tempLabelDirTest=fullfile(tempdir,sprintf('ResizedMasksTest_Fold%d',foldNum));
if isfolder(tempImageDirTest),rmdir(tempImageDirTest,'s');end
if isfolder(tempLabelDirTest),rmdir(tempLabelDirTest,'s');end
mkdir(tempImageDirTest);
mkdir(tempLabelDirTest);

for i=1:numel(resizedImagesTest)
imwrite(resizedImagesTest{i},fullfile(tempImageDirTest,sprintf('testImage_%d.png',i)));
imwrite(resizedMasksTest{i},fullfile(tempLabelDirTest,sprintf('testMask_%d.png',i)));
end

imdsTestResized=imageDatastore(tempImageDirTest);
pxdsTestResized=pixelLabelDatastore(tempLabelDirTest,classNames,labelIDs);

metrics=evaluateModel(net,pxdsTestResized,imdsTestResized,foldNum,segResultsRoot,statsRoot,origImagePaths); 
classifyTissueTypes(pxdsTestResized,classNames,tissueRoot,foldNum);

rmdir(tempImageDirTest,'s');
rmdir(tempLabelDirTest,'s');
else
fprintf('Model for Fold %d was not found: %s\n',foldNum,foldModelName);
end
end

function classifyTissueTypes(pxdsTestResized,classNames,resultsFolder,foldNum)
fprintf('Tissue type classification in ground-truth mask for Fold %d...\n',foldNum);
healthyClasses=categorical("background");
diseasedClasses=categorical("wound");
resultsTable=table('Size',[0,3],...
'VariableTypes',{'double','double','double'},...
'VariableNames',{'Image','Healthy tissue [%]','Pathological changes [%]'});
for i=1:numel(pxdsTestResized.Files)
mask=readimage(pxdsTestResized,i);
totalPixels=numel(mask);
healthyPixels=sum(ismember(mask(:),healthyClasses));
diseasedPixels=sum(ismember(mask(:),diseasedClasses));
healthyPercentage=(healthyPixels/totalPixels)*100;
diseasedPercentage=(diseasedPixels/totalPixels)*100;
resultsTable=[resultsTable;{i,healthyPercentage,diseasedPercentage}];
end
if ~isfolder(resultsFolder)
error('Output folder does not exist: %s',resultsFolder);
end
excelFile=fullfile(resultsFolder,sprintf('Fold_%d_Results_UNET_RGB_HSV_IR.xlsx',foldNum));
fprintf('Saving tissue classification results to file: %s\n',excelFile);
writetable(resultsTable,excelFile,'Sheet','TissueClassification');
fprintf('Tissue classification summary for Fold %d has been saved to the Excel file: %s\n',foldNum,excelFile);
end

function [syncedImages,syncedLabels]=synchronizeDatasets(images,labels,suffixRemoveImage,suffixRemoveLabel)
[~,imageNames,~]=cellfun(@fileparts,images,'UniformOutput',false);
[~,labelNames,~]=cellfun(@fileparts,labels,'UniformOutput',false);
imageNamesClean=strrep(imageNames,suffixRemoveImage,'');
labelNamesClean=strrep(labelNames,suffixRemoveLabel,'');

[commonNames,imgIdx,labelIdx]=intersect(imageNamesClean,labelNamesClean,'stable');
syncedImages=images(imgIdx);
syncedLabels=labels(labelIdx);
end

function imgRGB=ensureRGB(img)
if size(img,3)==1
imgRGB=repmat(img,[1,1,3]);
else
imgRGB=img;
end
end

function metrics=evaluateModel(net,pxdsTestResized,imdsTestResized,foldNum,segResultsRoot,statsRoot,origImagePaths)
fprintf('Starting semantic segmentation (UNET RGB-HSV-IR)...\n');
pxdsResults=semanticseg(imdsTestResized,net,'MiniBatchSize',8);
visualizeSegmentationResults(net,imdsTestResized,pxdsTestResized,pxdsResults,segResultsRoot,foldNum);
woundMaskDir=fullfile(segResultsRoot,'WoundMasks',sprintf('Fold%d',foldNum));
if ~isfolder(woundMaskDir),mkdir(woundMaskDir);end
savePredictedMasks(pxdsResults,woundMaskDir);
metrics=calculateMetrics(pxdsTestResized,pxdsResults,foldNum,statsRoot,origImagePaths);
end

function savePredictedMasks(pxdsResults,outputDir)
fprintf('Saving predicted masks to folder: %s\n',outputDir);
for i=1:numel(pxdsResults.Files)
predictedMask=readimage(pxdsResults,i);
binaryMask=uint8(predictedMask=='wound');
maskBW=uint8(binaryMask)*255;
imwrite(maskBW,fullfile(outputDir,sprintf('predictedMask_%d.png',i)));
end
end

function visualizeSegmentationResults(net,imdsTestResized,pxdsTestResized,pxdsResults,segResultsRoot,foldNum)
fprintf('Visualizing segmentation results for all images (UNET)...\n');
hFig=figure(1);
numFiles=numel(imdsTestResized.Files);
alpha=0.5;
colorWound=[1 0 0];
for k=1:numFiles
testImage=readimage(imdsTestResized,k);
trueMaskCat=readimage(pxdsTestResized,k);
predictedMask=readimage(pxdsResults,k);
trueWoundMask=(trueMaskCat=='wound');
predWoundMask=(predictedMask=='wound');
trueOverlay=imoverlay_alpha(testImage,trueWoundMask,colorWound,alpha);
predictedOverlay=imoverlay_alpha(testImage,predWoundMask,colorWound,alpha);
figure(hFig);
subplot(1,3,1);
imshow(testImage);
title('Original Image');
subplot(1,3,2);
imshow(trueOverlay);
title('Ground Truth Mask');
subplot(1,3,3);
imshow(predictedOverlay);
title('Predicted Mask');
outFold=fullfile(segResultsRoot,'Visualization_Results',sprintf('Fold%d',foldNum));
if ~isfolder(outFold),mkdir(outFold);end
saveas(hFig,fullfile(outFold,sprintf('Result_%d.png',k)));
drawnow;
fprintf('Processed image %d/%d.\n',k,numFiles);
pause(1);
end
if isvalid(hFig)
close(hFig);
end
fprintf('Visualization has been saved in: %s\n',fullfile(segResultsRoot,'Visualization_Results',sprintf('Fold%d',foldNum)));
end

function metrics=calculateMetrics(pxdsTrue,pxdsPred,foldNum,statsRoot,origImagePaths)
fprintf('Computing metrics for Fold %d...\n',foldNum);
nTrue=numel(pxdsTrue.Files);
nPaths=numel(origImagePaths);
if nTrue~=nPaths
warning('Fold %d: the number of masks (pxdsTrue=%d) differs from the number of paths (origImagePaths=%d).',foldNum,nTrue,nPaths);
end
totalTP=0;totalTN=0;totalFP=0;totalFN=0;
totalPixelCounts=zeros(1,2);
bgPix=zeros(nTrue,1);
woundPix=zeros(nTrue,1);
totalPix=zeros(nTrue,1);
bgPercent=zeros(nTrue,1);
woundPercent=zeros(nTrue,1);
TP_img=zeros(nTrue,1);
TN_img=zeros(nTrue,1);
FP_img=zeros(nTrue,1);
FN_img=zeros(nTrue,1);
for i=1:nTrue
trueMask=readimage(pxdsTrue,i);
predMask=readimage(pxdsPred,i);
trueBinary=(trueMask=='wound');
predBinary=(predMask=='wound');
TP=sum((trueBinary(:)==1)&(predBinary(:)==1));
TN=sum((trueBinary(:)==0)&(predBinary(:)==0));
FP=sum((trueBinary(:)==0)&(predBinary(:)==1));
FN=sum((trueBinary(:)==1)&(predBinary(:)==0));
TP_img(i)=TP;
TN_img(i)=TN;
FP_img(i)=FP;
FN_img(i)=FN;
totalTP=totalTP+TP;
totalTN=totalTN+TN;
totalFP=totalFP+FP;
totalFN=totalFN+FN;
bg=sum(trueMask(:)=='background');
wd=sum(trueMask(:)=='wound');
tot=bg+wd;
bgPix(i)=bg;
woundPix(i)=wd;
totalPix(i)=tot;
bgPercent(i)=100*bg/max(tot,1);
woundPercent(i)=100*wd/max(tot,1);
totalPixelCounts(1)=totalPixelCounts(1)+bg;
totalPixelCounts(2)=totalPixelCounts(2)+wd;
end
classNames=["background","wound"];
totalPixels=sum(totalPixelCounts);
pixelPercentages=(totalPixelCounts/max(totalPixels,1))*100;
foldColumn=repmat(foldNum,numel(classNames),1);
pixelCountsTable=table(foldColumn,classNames',pixelPercentages',...
'VariableNames',{'NumFold','Class','Number of pixels [%]'});
imgPathsCol=cell(nTrue,1);
for i=1:nTrue
if i<=nPaths
imgPathsCol{i}=origImagePaths{i};
else
imgPathsCol{i}="";
end
end
foldColImg=repmat(foldNum,nTrue,1);
perImageTable=table(foldColImg,imgPathsCol,bgPix,woundPix,...
bgPercent,woundPercent,totalPix,TP_img,TN_img,FP_img,FN_img,...
'VariableNames',{'Fold','ImagePath',...
'BackgroundPixels','WoundPixels',...
'BackgroundPercent','WoundPercent',...
'TotalPixels','TP','TN','FP','FN'});
excelFileName=fullfile(statsRoot,'SegmentationMetrics_UNET_DICE_LOSS_RGB_HSV_IR.xlsx');
pixelCountSheet='Weights_Class Pixels';
metricsSheetName='Metrics';
totalsSheetName='Sum_TP_TN_FP_FN';
perImageSheetName='ImageStatistics';
Validation="YES";
Activation_function="Softmax";
Loss_function="DiceLoss";
Optimizer="ADAM";
Precision=totalTP/(totalTP+totalFP+eps);
Sensitivity=totalTP/(totalTP+totalFN+eps);
Specificity=totalTN/(totalTN+totalFP+eps);
Accuracy=(totalTP+totalTN)/(totalTP+totalTN+totalFP+totalFN+eps);
FMeasure=2*(Precision*Sensitivity)/(Precision+Sensitivity+eps);
DiceScore=2*totalTP/(2*totalTP+totalFP+totalFN+eps);
IoU=totalTP/(totalTP+totalFP+totalFN+eps);
MCC=((totalTP*totalTN)-(totalFP*totalFN))/sqrt((totalTP+totalFP)*(totalTP+totalFN)*(totalTN+totalFP)*(totalTN+totalFN)+eps);
AUC=(Sensitivity+Specificity)/2;
metricsTable=table(foldNum,Validation,Activation_function,...
Loss_function,Optimizer,Accuracy,Precision,...
Sensitivity,Specificity,FMeasure,DiceScore,IoU,...
MCC,AUC,...
'VariableNames',{'Fold','Validation','Activation function',...
'Loss function','Optimizer','Accuracy','Precision',...
'Sensitivity','Specificity','FMeasure','DiceScore',...
'IoU','MCC','AUC'});
totalsTable=table(foldNum,totalTP,totalTN,totalFP,totalFN,...
'VariableNames',{'Fold','TotalTP','TotalTN','TotalFP','TotalFN'});
existingMetrics=table();
existingTotals=table();
existingPerImg=table();
existingPixelCounts=table();
if isfile(excelFileName)
[~,sheets]=xlsfinfo(excelFileName);
if ismember(metricsSheetName,sheets)
existingMetrics=readtable(excelFileName,'Sheet',metricsSheetName,'VariableNamingRule','preserve');
metricsTable=[existingMetrics;metricsTable];
end
if ismember(totalsSheetName,sheets)
existingTotals=readtable(excelFileName,'Sheet',totalsSheetName,'VariableNamingRule','preserve');
totalsTable=[existingTotals;totalsTable];
end
if ismember(perImageSheetName,sheets)
existingPerImg=readtable(excelFileName,'Sheet',perImageSheetName,'VariableNamingRule','preserve');
neededVars={'Fold','ImagePath','BackgroundPixels','WoundPixels',...
'BackgroundPercent','WoundPercent','TotalPixels',...
'TP','TN','FP','FN'};
if all(ismember(neededVars,existingPerImg.Properties.VariableNames))&&width(existingPerImg)==numel(neededVars)
perImageTable=[existingPerImg;perImageTable];
else
warning('Existing sheet %s has a different structure – it will be overwritten with a new table.',perImageSheetName);
end
end
if ismember(pixelCountSheet,sheets)
existingPixelCounts=readtable(excelFileName,'Sheet',pixelCountSheet,'VariableNamingRule','preserve');
pixelCountsTable=[existingPixelCounts;pixelCountsTable];
end
end
writetable(metricsTable,excelFileName,'Sheet',metricsSheetName);
writetable(pixelCountsTable,excelFileName,'Sheet',pixelCountSheet);
writetable(totalsTable,excelFileName,'Sheet',totalsSheetName);
writetable(perImageTable,excelFileName,'Sheet',perImageSheetName);
fprintf('Metrics have been saved to the file: %s\n',excelFileName);
metrics=table2struct(metricsTable);
end

function out=imoverlay_alpha(img,mask,color,alpha)
img=im2double(img);
out=img;
for c=1:3
channel=out(:,:,c);
channel(mask)=(1-alpha)*channel(mask)+alpha*color(c);
out(:,:,c)=channel;
end
out=im2uint8(out);
end

function maskOut=normalizeGT(maskIn)
if size(maskIn,3)>1
maskIn=maskIn(:,:,1);
end
mask=double(maskIn);
u=unique(mask(:));
bgLabel=1;
woundLabel=2; 
if numel(u)==1
maskStd=ones(size(mask))*bgLabel;
else
bgVal=min(u);
woundVal=max(u);
maskStd=zeros(size(mask),'uint8');
maskStd(mask==bgVal)=bgLabel;
maskStd(mask==woundVal)=woundLabel;
other=(mask~=bgVal)&(mask~=woundVal);
maskStd(other)=woundLabel;
end
maskOut=uint8(maskStd);
end
