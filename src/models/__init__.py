import torch

from src.models.cnn import SimpleCNN
from src.models.transfer import build_transfer


def build_model(name, num_classes, pretrained=True):
    if name == "cnn":
        return SimpleCNN(num_classes)
    return build_transfer(name, num_classes, pretrained=pretrained)


def get_gradcam_layer(model, name):
    """Tầng tích chập cuối dùng cho Grad-CAM."""
    if name == "resnet50":
        return model.layer4[-1]
    return model.features[-1]  # cnn, mobilenet_v2, efficientnet_b0


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
