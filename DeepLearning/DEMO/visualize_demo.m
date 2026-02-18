clear; clc; close all;

rootFolder ='C:/Users/korne/Desktop/DEMO';  

modelFile = fullfile(rootFolder,'deeplabv3plus_resnet50_case_6_1_1_demo.mat');
imgPath = fullfile(rootFolder,'case_6_1_1_rgb.png');

classNames = ["background","wound"];
labelIDs = [1 2];
inputSize = [224 224 3];

S = load(modelFile);  
net = S.net;

I = readAndPreprocessImage(imgPath,inputSize);
C = semanticseg(I,net);

predMask = uint8(C);
predMask(C == "background") = 1;
predMask(C == "wound") = 2;

B = labeloverlay(I, C,'IncludedLabels',"wound",'Transparency',0.5);
overlayFile = fullfile(rootFolder,'case_6_1_1_overlay.png');
imwrite(B,overlayFile);
fprintf('Overlay was saved in: %s\n',overlayFile);

figure;
imshow(B);
title('Prediction');

maskWound = (C == "wound");
predMaskGray = uint8(maskWound) * 255;
imwrite(predMaskGray, fullfile(rootFolder,'case_6_1_1_prediction_mask.png'));

function I = readAndPreprocessImage(filename, inputSize)
    I = imread(filename);
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    I = imresize(I, inputSize(1:2));
end
