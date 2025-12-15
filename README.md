# Wound-AI-Deep-Learning-for-Chronic-Wound-Detection-and-Segmentation

1. TRAINING folder:
The TRAINING directory contains MATLAB scripts used to train all segmentation networks (DeepLabV3+, SegNet and U-Net) for
different image modalities (RGB, RGB+HSV+IR). Each training script saves the trained network (net) and optional training information (info) into the MODELS directory as a .mat file (one file per fold).

2. MODELS folder:
The MODELS directory stores all trained segmentation models produced by the scripts in TRAINING.
**The MODELS directory is created automatically during the training process**:
**The scripts in the TRAINING folder create this directory along with the corresponding subfolders for each architecture and modality.**
For each architecture and modality there is a dedicated subfolder:
- MODELS/DEEPLAB_ResNet50_RGB_SGDM - DeepLabV3+ (ResNet-50 backbone) trained on RGB images with SGDM optimizer.
- MODELS/DEEPLAB_ResNet50_RGB_ADAM - DeepLabV3+ (ResNet-50 backbone) trained on RGB images with ADAM optimizer.
- MODELS/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM - DeepLabV3+ (ResNet-50 backbone) trained on RGB + HSV + IR channels with ADAM optimizer.
- MODELS/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM - DeepLabV3+ (ResNet-50 backbone) trained on RGB + HSV + IR channels with SGDM optimizer.
- MODELS/SEGNET_VGG19_RGB - SegNet (VGG-19 backbone) trained on RGB images with ADAM optimizer.
- MODELS/SEGNET_VGG19_RGB_HSV_IR - SegNet with VGG-19 encoder trained on  RGB + HSV + IR channels with ADAM optimizer.
- MODELS/UNET_DICE_LOSS_RGB - U-Net trained on RGB images with Dice Loss.
- MODELS/UNET_DICE_LOSS_RGB_HSV_IR - U-Net trained on RGB + HSV + IR channels with Dice Loss.

3. MODELS_EVALUATIONS folder:
The MODELS_EVALUATIONS directory contains MATLAB scripts used to evaluate and visualize the trained models from MODELS.
**For each model family there is a dedicated subfolder:
- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_SGDM,
- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_ADAM,
- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM,
- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM,
- MODELS_EVALUATIONS/SEGNET_VGG19_RGB,
- MODELS_EVALUATIONS/SEGNET_VGG19_RGB_HSV_IR,
- MODELS_EVALUATIONS/UNET_DICE_LOSS_RGB,
- MODELS_EVALUATIONS/UNET_DICE_LOSS_RGB_HSV_IR.**

4. GradCAM folder:
The GradCAM directory stores Gradient-weighted Class Activation Map (Grad-CAM) visualizations generated for the trained models, helping to interpret
which image regions contributed most to the model’s predictions. The trained models are always loaded directly from the corresponding subfolders inside the **MODELS** directory.
For each model family there is a dedicated subfolder:
- GradCAM/DEEPLAB_ResNet50_RGB_SGDM,
- GradCAM/DEEPLAB_ResNet50_RGB_ADAM,
- GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM,
- GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM,
- GradCAM/SEGNET_VGG19_RGB,
- GradCAM/SEGNET_VGG19_RGB_HSV_IR,
- GradCAM/UNET_DICE_LOSS_RGB,
- GradCAM/UNET_DICE_LOSS_RGB_HSV_IR.
Each subfolder includes:
- gradCAM_segmentation.m - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image;
- gradcam_deeeplabv3Plus_rgb_validation.m – the main script that should be executed.
