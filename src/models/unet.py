"""U-Net with a pretrained ResNet-34 encoder, adapted to multispectral input."""
import segmentation_models_pytorch as smp

def build_unet(in_channels=6, classes=1, encoder="resnet34", encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
