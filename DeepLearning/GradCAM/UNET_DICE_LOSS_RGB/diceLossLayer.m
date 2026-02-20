classdef diceLossLayer < nnet.layer.ClassificationLayer

    methods
        function layer = diceLossLayer(name)
            layer.Name = name;
            layer.Description = 'Dice Loss Layer';
        end
        
        function loss = forwardLoss(layer, Y, T)
            T = single(T);
            Y = single(Y);
            intersection = sum(sum(Y .* T, 1), 2);
            union = sum(sum(Y, 1), 2) + sum(sum(T, 1), 2);
            diceCoefficient = (2 * intersection) ./ (union + 1e-6);
            loss = single(1 - mean(diceCoefficient, 'all')); 
        end
    end
end
