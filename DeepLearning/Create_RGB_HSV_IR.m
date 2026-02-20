clear; clc; close all;

rootFolder = 'D:/DeepLearning'; % OWN PATH WITH FOLDS 1-10
outputFolder = 'D:/DeepLearning'; % SAME PATH AS ABOVE

numFolds = 10;

for foldNum = 1:numFolds
    fprintf('\nProcessing Fold %d...\n', foldNum);

    currentFoldPath = fullfile(rootFolder, sprintf('Fold%d', foldNum));
    outputFoldPath = fullfile(outputFolder, sprintf('Fold%d', foldNum));

    if ~isfolder(outputFoldPath)
        mkdir(outputFoldPath);
    end

    caseDirs = dir(fullfile(currentFoldPath, 'case_*'));
    for caseNum = 1:numel(caseDirs)
        casePath = fullfile(caseDirs(caseNum).folder, caseDirs(caseNum).name);

        rgbFile = dir(fullfile(casePath, '*_rgb.png'));
        hsvFile = dir(fullfile(casePath, '*_hsv.tiff'));

        irFilePng = dir(fullfile(casePath, '*_ir.png'));
        irFileTiff = dir(fullfile(casePath, '*_ir.tiff'));

        if ~isempty(rgbFile) && ~isempty(hsvFile) && (~isempty(irFilePng) || ~isempty(irFileTiff))
            rgbImage = imread(fullfile(rgbFile.folder, rgbFile.name));
            hsvImage = imread(fullfile(hsvFile.folder, hsvFile.name));

            if ~isempty(irFilePng)
                irImage = imread(fullfile(irFilePng.folder, irFilePng.name));
            else
                irImage = imread(fullfile(irFileTiff.folder, irFileTiff.name));
            end

            rgbGray = im2gray(rgbImage);
            grayHSV = im2gray(hsv2rgb(hsvImage / 255));
            irGray = im2gray(irImage);

            combinedImage = cat(3, rgbGray, grayHSV, irGray);
            
            outputCasePath = fullfile(outputFoldPath, caseDirs(caseNum).name);
            if ~isfolder(outputCasePath)
                mkdir(outputCasePath);
            end

            outputFileName = sprintf('%s_obraz.png', caseDirs(caseNum).name);
            imwrite(combinedImage, fullfile(outputCasePath, outputFileName));
        else
            fprintf('Missing required files in folder: %s\n', casePath);
        end
    end
end

fprintf('Processing completed. All images saved in folder: %s\n', outputFolder);

%%Example visualization
figure;
subplot(1, 3, 1); 
imshow(rgbGray, []); 
title('RGB channel (grayscale)');
subplot(1, 3, 2); 
imshow(grayHSV, []); 
title('HSV channel');
subplot(1, 3, 3); 
imshow(irGray, []); 
title('IR channel');
