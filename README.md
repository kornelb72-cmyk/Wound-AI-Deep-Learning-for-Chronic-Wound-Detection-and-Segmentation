# Wound-AI-Deep-Learning-for-Chronic-Wound-Detection-and-Segmentation

**TRAINING folder**:<br>

The TRAINING directory contains MATLAB scripts used to train all segmentation networks (DeepLabV3+, SegNet and U-Net) for
different image modalities (RGB, RGB + HSV + IR). Each training script saves the trained network (net) and optional training information (info) into the **MODELS** directory as a **.mat file** (one file per fold).

For **UNET_DICE_LOSS_RGB and UNET_DICE_LOSS_RGB_HSV_IR** specifically, the subfolders contain:<br>
**- main scripts: train_save_model_UNET_RGB.m and train_save_model_UNET_RGB_HSV_IR.m** - the main training script for the U-Net with Dice loss that should be executed,<br>
**- helpers: diceLossLayer.m** - a custom MATLAB layer that implements the Dice Loss function and is required by the U-Net Dice-Loss models during training.


**MODELS folder**:<br>

The MODELS directory stores all trained segmentation models produced by the scripts in **TRAINING** folder.<br>
**The MODELS directory is created automatically during the training process**:<br>
**The scripts in the TRAINING folder create this directory along with the corresponding subfolders for each architecture and modality.**

**For each architecture and modality there is a dedicated subfolder**:<br>
**- MODELS/DEEPLAB_ResNet50_RGB_SGDM** - DeepLabV3+ (ResNet-50 backbone) trained on RGB images with SGDM optimizer.<br>
**- MODELS/DEEPLAB_ResNet50_RGB_ADAM** - DeepLabV3+ (ResNet-50 backbone) trained on RGB images with ADAM optimizer.<br>
**- MODELS/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM** - DeepLabV3+ (ResNet-50 backbone) trained on RGB + HSV + IR channels with ADAM optimizer.<br>
**- MODELS/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM** - DeepLabV3+ (ResNet-50 backbone) trained on RGB + HSV + IR channels with SGDM optimizer.<br>
**- MODELS/SEGNET_VGG19_RGB** - SegNet (VGG-19 backbone) trained on RGB images with ADAM optimizer.<br>
**- MODELS/SEGNET_VGG19_RGB_HSV_IR** - SegNet with VGG-19 encoder trained on  RGB + HSV + IR channels with ADAM optimizer.<br>
**- MODELS/UNET_DICE_LOSS_RGB**- U-Net trained on RGB images with with a custom MATLAB Dice Loss layer.<br>
**- MODELS/UNET_DICE_LOSS_RGB_HSV_IR** - U-Net trained on RGB + HSV + IR channels with a custom MATLAB Dice Loss layer.<br>

**MODELS_EVALUATIONS folder**:<br>

The MODELS_EVALUATIONS directory contains MATLAB scripts used to evaluate and visualize the trained models from **MODELS**.<br>

**For each model family there is a dedicated subfolder**:<br>
**- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_SGDM**,<br>
**- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_ADAM**,<br>
**- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM**,<br>
**- MODELS_EVALUATIONS/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM**,<br>
**- MODELS_EVALUATIONS/SEGNET_VGG19_RGB**,<br>
**- MODELS_EVALUATIONS/SEGNET_VGG19_RGB_HSV_IR**,<br>
**- MODELS_EVALUATIONS/UNET_DICE_LOSS_RGB**,<br>
**- MODELS_EVALUATIONS/UNET_DICE_LOSS_RGB_HSV_IR**.<br>

**GradCAM folder**:<br>

The GradCAM directory stores Gradient-weighted Class Activation Map (Grad-CAM) visualizations generated for the trained models, helping to interpret which image regions contributed most to the model’s predictions. The trained models are always loaded directly from the corresponding subfolders inside the **MODELS** directory.<br>

**For each model family there is a dedicated subfolder**:<br>
**- helper: GradCAM/DEEPLAB_ResNet50_RGB_ADAM/gradCAM_segmentation.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/DEEPLAB_ResNet50_RGB_ADAM/gradcam_deeeplabv3Plus_rgb_validation.m** - the main script that should be executed.<br>
   
**- helper: GradCAM/DEEPLAB_ResNet50_RGB_SGDM/gradCAM_segmentation.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/DEEPLAB_ResNet50_RGB_SGDM/gradcam_deeeplabv3Plus_sgdm_validation.m** - the main script that should be executed.<br>

**- helper: GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM/gradCAM_segmentation.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_ADAM/gradcam_deeeplabv3Plus_rgb_hsv_ir_validation.m** - the main script that should be executed.<br>

**- helper: GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM/gradCAM_segmentation.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/DEEPLAB_ResNet50_RGB_HSV_IR_SGDM/gradcam_deeeplabv3Plus_rgb_hsv_ir_sgdm_validation.m** - the main script that should be executed.<br>

**- helper: GradCAM/SEGNET_VGG19_RGB/gradCAM_segnet.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/SEGNET_VGG19_RGB/gradcam_segnet_vgg19_validation.m** - the main script that should be executed.<br>

**- helper: GradCAM/SEGNET_VGG19_RGB_HSV_IR/gradCAM_segnet.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/SEGNET_VGG19_RGB_HSV_IR/gradcam_segnet_vgg19_RGB_HSV_IR_validation.m** - the main script that should be executed.<br>

**- helper: GradCAM/UNET_DICE_LOSS_RGB/diceLossLayer.m** - a custom MATLAB layer that computes the Dice loss for the segmentation output and the ground-truth mask,<br>
**- helper: GradCAM/UNET_DICE_LOSS_RGB/gradCAM_unet.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/UNET_DICE_LOSS_RGB/GradCAM_unet_dice_loss_rgb.m** - the main script that should be executed.<br>

**- helper: GradCAM/UNET_DICE_LOSS_RGB_HSV_IR/diceLossLayer.m** - a custom MATLAB layer that computes the Dice loss for the segmentation output and the ground-truth mask,<br>
**- helper: GradCAM/UNET_DICE_LOSS_RGB_HSV_IR/gradCAM_unet.m** - a helper function that computes Grad-CAM heatmaps for a given trained segmentation network and input image,<br>
**- main script: GradCAM/UNET_DICE_LOSS_RGB_HSV_IR/GradCAM_unet_dice_loss_RGB_HSV_IR.m** - the main script that should be executed.

