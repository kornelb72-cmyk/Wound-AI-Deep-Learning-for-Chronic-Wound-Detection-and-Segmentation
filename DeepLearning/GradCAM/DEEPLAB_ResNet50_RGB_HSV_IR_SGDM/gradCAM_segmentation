function [cam, featureLayerName, scoreLayerName] = gradCAM_segmentation(net, I, classIdx)

featureLayerName = 'activation_49_relu';   
scoreLayerName = 'scorer';              

lgraph = layerGraph(net);
layerNames = {lgraph.Layers.Name};

if any(strcmp(layerNames, 'pixelOutput'))
    lgraph = removeLayers(lgraph, 'pixelOutput');
end

dlnet = dlnetwork(lgraph);

I = im2single(I);
if size(I,3) == 1
    I = repmat(I,1,1,3);
end
dlX = dlarray(I,'SSCB');

[loss, gradients, dlFeatures, classScoreMap] = dlfeval(@modelGradientsSeg,dlnet, dlX, classIdx, featureLayerName, scoreLayerName);

scoreMin = min(extractdata(classScoreMap),[],'all');
scoreMax = max(extractdata(classScoreMap),[],'all');
gradNorm = norm(extractdata(gradients(:)));

fprintf('Grad-CAM debug: scoreMin=%.4f scoreMax=%.4f, gradNorm=%.4e\n',scoreMin, scoreMax, gradNorm);

gradientsData = extractdata(gradients);    
featuresData  = extractdata(dlFeatures); 

gradientsData = gradientsData(:,:,:,1);
featuresData  = featuresData(:,:,:,1);

alpha = squeeze(mean(mean(gradientsData,1),2)); 

[Hf,Wf,K] = size(featuresData);
cam = zeros(Hf,Wf,'single');
for k = 1:K
    cam = cam + alpha(k) * featuresData(:,:,k);
end

cam = max(cam,0);
if max(cam(:)) > 0
    cam = cam - min(cam(:));
    cam = cam / (max(cam(:)) + eps);
end

[H,W,~] = size(I);
cam = imresize(cam,[H W]);
cam = mat2gray(cam);

end

function [loss, gradients, features, classScoreMap] = modelGradientsSeg(dlnet, dlX, classIdx, featureLayerName, scoreLayerName)

[dlScores, features] = forward(dlnet, dlX,'Outputs',{scoreLayerName, featureLayerName});

classScoreMap = dlScores(:,:,classIdx,1);

loss = mean(classScoreMap,'all');

gradients = dlgradient(loss, features);

end
