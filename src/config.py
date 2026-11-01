"""Cấu hình dùng chung cho toàn bộ project."""

# 10 lớp cử chỉ lấy từ HaGRID (tên lớp đúng như trong file annotation của HaGRID)
CLASSES = ["call", "dislike", "fist", "like", "ok", "one", "palm", "peace", "rock", "stop"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

VI_NAMES = {
    "call": "Gọi điện", "dislike": "Dislike", "fist": "Nắm tay", "like": "Like",
    "ok": "OK", "one": "Một ngón", "palm": "Xòe tay", "peace": "Chữ V",
    "rock": "Rock", "stop": "Dừng",
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# HOG cho mô hình truyền thống (ảnh nhỏ hơn để vector đặc trưng không quá lớn)
HOG_SIZE = 128

SEED = 42

# Oẳn tù tì: cử chỉ -> nước đi
RPS_MAP = {"fist": "rock", "palm": "paper", "peace": "scissors"}

DL_MODELS = ["cnn", "mobilenet_v2", "efficientnet_b0", "resnet50"]
CLASSICAL_MODELS = ["svm", "rf", "knn"]
