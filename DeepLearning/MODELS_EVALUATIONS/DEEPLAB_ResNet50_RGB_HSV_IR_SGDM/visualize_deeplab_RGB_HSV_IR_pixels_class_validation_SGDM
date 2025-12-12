clear; clc; close all;

rootFolder = 'D:/DeepLearning';  %% OWN PATH


evalRoot = fullfile(rootFolder,'MODELS_EVALUATIONS','DEEPLAB_ResNet50_RGB_HSV_IR_SGDM');
if ~isfolder(evalRoot), mkdir(evalRoot); end

if ~isfolder(evalRoot), mkdir(evalRoot); end

modelsRoot = fullfile(rootFolder,'MODELS','DEEPLAB_ResNet50_RGB_HSV_IR_SGDM');
if ~isfolder(modelsRoot)
error('Models folder does not exist: %s',modelsRoot);
end


numFolds = 10;
classNames = ["background","wound"];
labelIDs = [1 2];
inputSize = [224 224 3];
maxEpochs = 50; 
initialLearnRate = 1e-4; 
L2Regularization = 1e-4; 

mainResultsRoot = fullfile(evalRoot,'Segmentation_Results_DEEPLABV3PLUS_RGB_HSV_IR_VALIDATION_SGDM');
if ~isfolder(mainResultsRoot), mkdir(mainResultsRoot); end
visRoot = fullfile(mainResultsRoot,'Visualization_Results');
if ~isfolder(visRoot), mkdir(visRoot); end
maskRoot = fullfile(mainResultsRoot,'WoundMasks');
if ~isfolder(maskRoot), mkdir(maskRoot); end
statsRoot = fullfile(mainResultsRoot,'Statistics');
if ~isfolder(statsRoot), mkdir(statsRoot); end
tissueRoot = fullfile(statsRoot,'TISSUE_CLASSIFICATION');
if ~isfolder(tissueRoot), mkdir(tissueRoot); end

for foldNum = 1:numFolds
fprintf('\nModel evaluation for Fold %d...\n', foldNum);
foldModelName = fullfile(modelsRoot,sprintf('deeplabv3plus_resnet50_validation_mixed_sgdm_fold_%d.mat', foldNum));
if isfile(foldModelName)
load(foldModelName,'net','info');
fprintf('Model for Fold %d has been loaded from %s.\n', foldNum, foldModelName);
fprintf('Preparing test data for Fold %d...\n', foldNum);
testFoldName = sprintf('Fold%d', foldNum);
testFoldPath = fullfile(rootFolder, testFoldName);
[imdsTestResized, pxdsTestResized, origImagePaths] = loadAndProcessTestData(testFoldPath, classNames, labelIDs, inputSize, foldNum);
fprintf('Starting evaluation for Fold %d...\n', foldNum);
metrics = evaluateModel(net, pxdsTestResized, imdsTestResized, foldNum, inputSize, classNames, origImagePaths, visRoot, maskRoot, statsRoot); 
classifyTissueTypes(pxdsTestResized, classNames, tissueRoot, foldNum);
else
fprintf('Model for Fold %d not found: %s\n', foldNum, foldModelName);
end
end

function classifyTissueTypes(pxdsTestResized, classNames, resultsFolder, foldNum) 
fprintf('Classification of tissue types in ground truth masks for Fold %d...\n', foldNum);
healthyClasses = categorical("background");
diseasedClasses = categorical("wound");
resultsTable = table('Size',[0 3],...
'VariableTypes',{'double','double','double'},...
'VariableNames',{'ImageIndex','HealthyTissuesPercent','PathologicalChangesPercent'});
for i = 1:numel(pxdsTestResized.Files)
mask = readimage(pxdsTestResized, i);
totalPixels = numel(mask);
healthyPixels = sum(ismember(mask(:), healthyClasses));
diseasedPixels = sum(ismember(mask(:), diseasedClasses));
healthyPercentage = (healthyPixels / totalPixels) * 100;
diseasedPercentage = (diseasedPixels / totalPixels) * 100;
resultsTable = [resultsTable; {i, healthyPercentage, diseasedPercentage}];
end
if ~isfolder(resultsFolder)
error('Output folder does not exist: %s', resultsFolder);
end
excelFile = fullfile(resultsFolder, sprintf('Fold_%d_TissueClassification.xlsx', foldNum));
fprintf('Saving tissue classification results to: %s\n', excelFile);
writetable(resultsTable, excelFile, 'Sheet', 'TissueClassification');
fprintf('Tissue classification summary for Fold %d has been saved to Excel file: %s\n', foldNum, excelFile);
end

function [imdsResized, pxdsResized, origImagePaths] = loadAndProcessTestData(testFoldPath, classNames, labelIDs, inputSize, foldNum)
fprintf('Loading and processing test data from: %s\n', testFoldPath);
imageFiles = dir(fullfile(testFoldPath, '**', '*_obraz.*'));
labelFiles = dir(fullfile(testFoldPath, '**', '*_gt.*'));
if isempty(imageFiles) || isempty(labelFiles)
error('No images or masks in folder: %s', testFoldPath);
end
imagePaths = fullfile({imageFiles.folder}, {imageFiles.name});
labelPaths = fullfile({labelFiles.folder}, {labelFiles.name});
[imagePathsSynced, labelPathsSynced] = synchronizeDatasets(imagePaths, labelPaths, '_obraz', '_gt');
if isempty(imagePathsSynced) || isempty(labelPathsSynced)
error('No matching images and masks found in folder: %s', testFoldPath);
end
origImagePaths = imagePathsSynced(:);
tempImageDir = fullfile(tempdir, sprintf('ProcessedImages_Fold%d', foldNum));
tempLabelDir = fullfile(tempdir, sprintf('ProcessedMasks_Fold%d', foldNum));
if isfolder(tempImageDir), rmdir(tempImageDir,'s'); end
if isfolder(tempLabelDir), rmdir(tempLabelDir,'s'); end
mkdir(tempImageDir);
mkdir(tempLabelDir);
bgLabel = labelIDs(1);
woundLabel = labelIDs(2); 
for i = 1:numel(imagePathsSynced)
img = imread(imagePathsSynced{i});
imgResized = imresize(img, inputSize(1:2));
if size(imgResized,3) == 1
imgResized = repmat(imgResized,[1 1 3]);
end
imwrite(imgResized, fullfile(tempImageDir, sprintf('image_%d.png', i)));
mask = imread(labelPathsSynced{i});
if size(mask,3) > 1
mask = mask(:,:,1);
end
mask = double(mask);
u = unique(mask(:));
if numel(u) == 1
maskStd = ones(size(mask)) * bgLabel;
elseif numel(u) >= 2
bgVal = min(u);
wVal = max(u); 
maskStd = zeros(size(mask),'uint8');
maskStd(mask == bgVal) = bgLabel;
maskStd(mask == max(u)) = labelIDs(2);
otherIdx = mask ~= bgVal & mask ~= max(u);
maskStd(otherIdx) = labelIDs(2);
end
maskResized = imresize(maskStd, inputSize(1:2), 'nearest');
imwrite(maskResized, fullfile(tempLabelDir, sprintf('mask_%d.png', i)));
end
imdsResized = imageDatastore(tempImageDir);
pxdsResized = pixelLabelDatastore(tempLabelDir, classNames, labelIDs);
validateDataset(imdsResized, pxdsResized, inputSize);
end

function validateDataset(imds, pxds, inputSize)
fprintf('Validating image and mask sizes...\n');
for i = 1:numel(imds.Files)
img = imread(imds.Files{i});
if ~isequal(size(img, 1:2), inputSize(1:2))
error('Image %s has size [%d, %d], expected [%d, %d].', imds.Files{i}, size(img, 1), size(img, 2), inputSize(1), inputSize(2));
end
end
for i = 1:numel(pxds.Files)
mask = imread(pxds.Files{i});
if ~isequal(size(mask, 1:2), inputSize(1:2))
error('Mask %s has size [%d, %d], expected [%d, %d].', pxds.Files{i}, size(mask, 1), size(mask, 2), inputSize(1), inputSize(2));
end
if size(mask, 3) ~= 1
error('Mask %s has an invalid number of channels: %d. Expected 1.', pxds.Files{i}, size(mask, 3));
end
end
end

function metrics = evaluateModel(net, pxdsTestResized, imdsTestResized, foldNum, inputSize, classNames, origImagePaths, visRoot, maskRoot, statsRoot) 
fprintf('Starting semantic segmentation for Fold %d...\n', foldNum);
pxdsResults = semanticseg(imdsTestResized, net, 'MiniBatchSize', 8);
visualizeSegmentationResults(imdsTestResized, pxdsResults, pxdsTestResized, classNames, visRoot, foldNum);
woundMaskDir = fullfile(maskRoot, sprintf('Fold%d', foldNum));
if ~isfolder(woundMaskDir), mkdir(woundMaskDir); end
savePredictedMasks(pxdsResults, woundMaskDir);
metrics = calculateMetrics(pxdsTestResized, pxdsResults, foldNum, statsRoot, origImagePaths);
end

function savePredictedMasks(pxdsResults, outputDir)
fprintf('Saving predicted masks to folder: %s\n', outputDir);
for i = 1:numel(pxdsResults.Files)
predictedMask = readimage(pxdsResults, i);
maskBW = uint8(predictedMask == 'wound') * 255;
imwrite(maskBW, fullfile(outputDir, sprintf('predictedMask_%d.png', i)));
end
end

function visualizeSegmentationResults(imds, pxdsResults, pxdsTrue, classNames, visRoot, foldNum) 
fprintf('Visualization of segmentation results for all images in Fold %d...\n', foldNum);
numFiles = numel(imds.Files);
outputDir = fullfile(visRoot, sprintf('Fold%d', foldNum));
if ~isfolder(outputDir), mkdir(outputDir); end
for i = 1:numFiles
testImage = readimage(imds, i);
predictedMask = readimage(pxdsResults, i);
trueMask = readimage(pxdsTrue, i);
trueWoundMask = (trueMask == 'wound');
predictedWoundMask = (predictedMask == 'wound');
woundColor = [1 0 0];
alpha = 0.5;
trueOverlay = imoverlay_alpha(testImage, trueWoundMask, woundColor, alpha);
predictedOverlay = imoverlay_alpha(testImage, predictedWoundMask, woundColor, alpha);
figure;
subplot(1, 3, 1);
imshow(testImage);
title('Original Image');
subplot(1, 3, 2);
imshow(trueOverlay);
title('Ground Truth');
subplot(1, 3, 3);
imshow(predictedOverlay);
title('Prediction');
saveas(gcf, fullfile(outputDir, sprintf('Result_%d.png', i)));
close(gcf);
end
fprintf('Visualization results have been saved in folder: %s\n', outputDir);
end

function metrics = calculateMetrics(pxdsTrue, pxdsPred, foldNum, statsRoot, origImagePaths)
fprintf('Calculating metrics for Fold %d...\n', foldNum);
nTrue = numel(pxdsTrue.Files);
nPaths = numel(origImagePaths);
if nTrue ~= nPaths
warning('Fold %d: the number of masks (pxdsTrue=%d) differs from the number of paths (origImagePaths=%d).', foldNum, nTrue, nPaths);
end
totalTP = 0; totalTN = 0; totalFP = 0; totalFN = 0;
totalPixelCounts = zeros(1, 2);
bgPix = zeros(nTrue,1);
woundPix = zeros(nTrue,1);
totalPix = zeros(nTrue,1);
bgPercent = zeros(nTrue,1);
woundPercent = zeros(nTrue,1);
TP_img = zeros(nTrue,1);
TN_img = zeros(nTrue,1);
FP_img = zeros(nTrue,1);
FN_img = zeros(nTrue,1);
for i = 1:nTrue
trueMask = readimage(pxdsTrue, i);
predMask = readimage(pxdsPred, i);
trueBinary = (trueMask == 'wound');
predBinary = (predMask == 'wound');
TP = sum((trueBinary(:) == 1) & (predBinary(:) == 1));
TN = sum((trueBinary(:) == 0) & (predBinary(:) == 0));
FP = sum((trueBinary(:) == 0) & (predBinary(:) == 1));
FN = sum((trueBinary(:) == 1) & (predBinary(:) == 0));
TP_img(i) = TP;
TN_img(i) = TN;
FP_img(i) = FP;
FN_img(i) = FN;
totalTP = totalTP + TP;
totalTN = totalTN + TN;
totalFP = totalFP + FP;
totalFN = totalFN + FN;
bg = sum(trueMask(:) == 'background');
wd = sum(trueMask(:) == 'wound');
tot = bg + wd;
bgPix(i) = bg;
woundPix(i) = wd;
totalPix(i) = tot;
bgPercent(i) = 100 * bg / max(tot,1);
woundPercent(i) = 100 * wd / max(tot,1);
totalPixelCounts(1) = totalPixelCounts(1) + bg;
totalPixelCounts(2) = totalPixelCounts(2) + wd;
end
classNames = ["background","wound"];
totalPixels = sum(totalPixelCounts);
pixelPercentages = (totalPixelCounts / max(totalPixels,1)) * 100;
foldColumn = repmat(foldNum, numel(classNames), 1);
pixelCountsTable = table(foldColumn, classNames', pixelPercentages', ...
'VariableNames', {'Fold','Class','PixelPercentage'});
imgPathsCol = cell(nTrue,1);
for i = 1:nTrue
if i <= nPaths
imgPathsCol{i} = origImagePaths{i};
else
imgPathsCol{i} = "";
end
end
foldColImg = repmat(foldNum, nTrue, 1);
perImageTable = table(foldColImg, imgPathsCol, bgPix, woundPix, ...
bgPercent, woundPercent, totalPix, ...
TP_img, TN_img, FP_img, FN_img, ...
'VariableNames', {'Fold','ImagePath', ...
'BackgroundPixels','WoundPixels', ...
'BackgroundPercent','WoundPercent', ...
'TotalPixels','TP','TN','FP','FN'});
excelFileName = fullfile(statsRoot,'SegmentationMetrics_deeplabv3plus_resnet50_RGB_HSV_IR_SGDM.xlsx');
pixelCountSheet = 'ClassPixelWeights';
metricsSheetName = 'Metrics';
totalsSheetName = 'Sum_TP_TN_FP_FN';
perImageSheetName = 'ImageStatistics';
Validation = "YES";
ActivationFunction = "Softmax";
LossFunction = "CrossEntropy";
Optimizer = "SGDM";
Precision = totalTP / (totalTP + totalFP + eps);
Sensitivity = totalTP / (totalTP + totalFN + eps);
Specificity = totalTN / (totalTN + totalFP + eps);
Accuracy = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN + eps);
FMeasure = 2 * (Precision * Sensitivity) / (Precision + Sensitivity + eps);
DiceScore = 2 * totalTP / (2 * totalTP + totalFP + totalFN + eps);
IoU = totalTP / (totalTP + totalFP + totalFN + eps);
MCC = ((totalTP * totalTN) - (totalFP * totalFN)) / ...
sqrt((totalTP + totalFP) * (totalTP + totalFN) * ...
(totalTN + totalFP) * (totalTN + totalFN) + eps);
AUC = (Sensitivity + Specificity) / 2;
metricsTable = table(foldNum, Validation, ActivationFunction, ...
LossFunction, Optimizer, Accuracy, Precision, ...
Sensitivity, Specificity, FMeasure, DiceScore, IoU, ...
MCC, AUC, ...
'VariableNames', {'Fold','Validation','ActivationFunction', ...
'LossFunction','Optimizer','Accuracy','Precision', ...
'Sensitivity','Specificity','FMeasure','DiceScore', ...
'IoU','MCC','AUC'});
totalsTable = table(foldNum, totalTP, totalTN, totalFP, totalFN, ...
'VariableNames', {'Fold','TotalTP','TotalTN','TotalFP','TotalFN'});
existingMetrics = table();
existingTotals = table();
existingPerImg = table();
existingPixelCounts = table();
if isfile(excelFileName)
[~, sheets] = xlsfinfo(excelFileName);
if ismember(metricsSheetName, sheets)
existingMetrics = readtable(excelFileName,'Sheet',metricsSheetName,'VariableNamingRule','preserve');
metricsTable = [existingMetrics; metricsTable];
end
if ismember(totalsSheetName, sheets)
existingTotals = readtable(excelFileName,'Sheet',totalsSheetName,'VariableNamingRule','preserve');
totalsTable = [existingTotals; totalsTable];
end
if ismember(perImageSheetName, sheets)
existingPerImg = readtable(excelFileName,'Sheet',perImageSheetName,'VariableNamingRule','preserve');
neededVars = {'Fold','ImagePath','BackgroundPixels','WoundPixels', ...
'BackgroundPercent','WoundPercent','TotalPixels', ...
'TP','TN','FP','FN'};
if all(ismember(neededVars, existingPerImg.Properties.VariableNames)) && width(existingPerImg) == numel(neededVars)
perImageTable = [existingPerImg; perImageTable];
else
warning('Existing sheet %s has a different structure and will be overwritten with a new table.', perImageSheetName);
end
end
if ismember(pixelCountSheet, sheets)
existingPixelCounts = readtable(excelFileName,'Sheet',pixelCountSheet,'VariableNamingRule','preserve');
pixelCountsTable = [existingPixelCounts; pixelCountsTable];
end
end
writetable(metricsTable, excelFileName, 'Sheet', metricsSheetName);
writetable(pixelCountsTable, excelFileName, 'Sheet', pixelCountSheet);
writetable(totalsTable, excelFileName, 'Sheet', totalsSheetName);
writetable(perImageTable, excelFileName, 'Sheet', perImageSheetName);
fprintf('Metrics have been saved to file: %s\n', excelFileName);
metrics = table2struct(metricsTable);
end

function [imagesSynced, labelsSynced] = synchronizeDatasets(images, labels, suffixImage, suffixLabel)
[~, imageNames, ~] = cellfun(@fileparts, images, 'UniformOutput', false);
[~, labelNames, ~] = cellfun(@fileparts, labels, 'UniformOutput', false);
imageNamesClean = strrep(imageNames, suffixImage, '');
labelNamesClean = strrep(labelNames, suffixLabel, '');
[commonNames, imgIdx, lblIdx] = intersect(imageNamesClean, labelNamesClean, 'stable'); 
imagesSynced = images(imgIdx);
labelsSynced = labels(lblIdx);
if isempty(commonNames)
warning('No matching pairs of images and masks were found.');
end
end

function out = imoverlay_alpha(img, mask, color, alpha)
img = im2double(img);
out = img;
for c = 1:3
channel = out(:,:,c);
channel(mask) = (1-alpha)*channel(mask) + alpha*color(c);
out(:,:,c) = channel;
end
out = im2uint8(out);
end
