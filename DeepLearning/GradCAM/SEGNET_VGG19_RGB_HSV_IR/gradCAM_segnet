function [cam, featureLayerName, scoreLayerName] = gradCAM_segnet(net, I, classIdx)

featureLayerName = 'relu5_4';        
scoreLayerName ='decoder1_conv1'; 

lgraph = layerGraph(net);
layerNames = {lgraph.Layers.Name};

if any(strcmp(layerNames, 'pixelLabels'))
    lgraph = removeLayers(lgraph, 'pixelLabels');
end

if any(strcmp(layerNames, 'pixelOutput'))
    lgraph = removeLayers(lgraph, 'pixelOutput');
end

if any(strcmp(layerNames, 'softmax'))
    lgraph = removeLayers(lgraph, 'softmax');
end

dlnet = dlnetwork(lgraph);

I = im2single(I);
if size(I,3) == 1
    I = repmat(I,1,1,3);
end

dlX = dlarray(I, 'SSCB');  

[loss, gradients, dlFeatures, classScoreMap] = dlfeval(@modelGradientsSegnet,dlnet, dlX, classIdx, featureLayerName, scoreLayerName);

scoreMin = min(extractdata(classScoreMap),[],'all');
scoreMax = max(extractdata(classScoreMap),[],'all');
gradNorm = norm(extractdata(gradients(:)));
fprintf('SegNet Grad-CAM debug: scoreMin=%.4f scoreMax=%.4f gradNorm=%.4e\n',scoreMin, scoreMax, gradNorm);

gradientsData = extractdata(gradients);   
featuresData  = extractdata(dlFeatures);  

gradientsData = gradientsData(:,:,:,1);
featuresData  = featuresData(:,:,:,1);

alpha = squeeze(mean(mean(gradientsData,1),2)); 

[Hf, Wf, K] = size(featuresData);
cam = zeros(Hf, Wf, 'single');
for k = 1:K
    cam = cam + alpha(k) * featuresData(:,:,k);
end

cam = max(cam,0);
if max(cam(:)) > 0
    cam = cam - min(cam(:));
    cam = cam / (max(cam(:)) + eps);
end

[H, W, ~] = size(I);
cam = imresize(cam,[H W]);
cam = mat2gray(cam);

end

function [loss, gradients, features, classScoreMap] = modelGradientsSegnet(dlnet, dlX, classIdx, featureLayerName, scoreLayerName)

[dlScores, features] = forward(dlnet, dlX,'Outputs',{scoreLayerName, featureLayerName});

classScoreMap = dlScores(:,:,classIdx,1);

loss = mean(classScoreMap,'all');

gradients = dlgradient(loss, features);

end
