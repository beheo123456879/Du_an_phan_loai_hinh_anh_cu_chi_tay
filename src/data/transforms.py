from torchvision import transforms as T

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_eval_transforms(img_size=IMG_SIZE):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transforms(img_size=IMG_SIZE, augment=True):
    """augment=False dùng cho thực nghiệm ablation (không tăng cường dữ liệu)."""
    if not augment:
        return get_eval_transforms(img_size)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
