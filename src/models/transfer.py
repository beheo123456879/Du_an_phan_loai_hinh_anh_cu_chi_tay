"""Học chuyển giao: MobileNetV2, EfficientNet-B0, ResNet50 (pretrained ImageNet, torchvision)."""
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import models

HEAD_ATTR = {"mobilenet_v2": "classifier", "efficientnet_b0": "classifier", "resnet50": "fc"}

# Số khối cuối được mở băng khi fine-tuning
DEFAULT_UNFREEZE = {"mobilenet_v2": 5, "efficientnet_b0": 3, "resnet50": 1}


def build_transfer(name, num_classes, pretrained=True, dropout=0.2):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.last_channel, num_classes))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    else:
        raise ValueError(f"Mô hình không hỗ trợ: {name}")
    return m


def freeze_backbone(model, name):
    """Feature extraction: chỉ huấn luyện lớp phân loại cuối."""
    for p in model.parameters():
        p.requires_grad = False
    for p in getattr(model, HEAD_ATTR[name]).parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model, name, n_blocks=None):
    """Fine-tuning: mở băng n khối cuối của mạng nền."""
    n_blocks = n_blocks or DEFAULT_UNFREEZE[name]
    if name in ("mobilenet_v2", "efficientnet_b0"):
        blocks = list(model.features.children())[-n_blocks:]
    else:  # resnet50
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4][-n_blocks:]
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True


def freeze_bn_stats(model):
    """Giữ BatchNorm của phần bị đóng băng ở chế độ eval để không làm trôi thống kê ImageNet."""
    for m in model.modules():
        if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
