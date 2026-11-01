# Phân loại ảnh cử chỉ tay — Học máy

So sánh các mô hình phân loại 10 lớp cử chỉ tay (bộ dữ liệu HaGRID):

| Nhóm | Mô hình |
|---|---|
| Học máy truyền thống | HOG + **SVM**, HOG + **Random Forest**, HOG + **KNN** |
| Học sâu từ đầu | **CNN** tự thiết kế |
| Học chuyển giao | **MobileNetV2**, EfficientNet-B0, ResNet50 |

10 lớp: `call, dislike, fist, like, ok, one, palm, peace, rock, stop` (sửa trong `src/config.py`).

## Cấu trúc

```
hand_gesture_classification/
├── src/
│   ├── config.py              # danh sách lớp, kích thước ảnh, hằng số
│   ├── utils.py               # seed, device, json
│   ├── data/
│   │   ├── crop.py            # mở rộng bbox thành khung vuông
│   │   ├── prepare_hagrid.py  # cắt tay theo bbox HaGRID -> 224x224 + metadata.csv
│   │   ├── prepare_custom.py  # chuẩn bị tập tự thu (test ngoài phân phối)
│   │   ├── split.py           # chia 70/15/15 phân tầng, không trùng người
│   │   ├── transforms.py      # tăng cường dữ liệu / chuẩn hóa ImageNet
│   │   └── dataset.py         # Dataset + DataLoader
│   ├── models/
│   │   ├── classical.py       # HOG + SVM / RF / KNN
│   │   ├── cnn.py             # CNN tự thiết kế (4 khối Conv-BN-ReLU-MaxPool)
│   │   └── transfer.py        # MobileNetV2, EfficientNet-B0, ResNet50 + đóng băng / fine-tune
│   ├── train_classical.py     # Grid Search + kiểm định chéo cho SVM / RF / KNN
│   ├── train_dl.py            # huấn luyện CNN / MobileNetV2 / ...
│   ├── evaluate.py            # Accuracy, Precision, Recall, F1 macro, ma trận nhầm lẫn
│   ├── benchmark.py           # số tham số, kích thước, thời gian suy luận CPU
│   ├── gradcam.py             # Grad-CAM
│   ├── compare_results.py     # bảng + biểu đồ so sánh mọi mô hình
│   └── hand_detector.py       # MediaPipe Hands (chỉ dùng cho demo webcam)
├── app/                       # demo webcam (chạy trên máy cá nhân, không cần cho Kaggle)
└── requirements.txt
```

Mỗi mô hình lưu kết quả vào `outputs/<tên>/`: `best.pt` hoặc `model.joblib`, `results.json`,
`cm_test.png` (ma trận nhầm lẫn), `history.png` (đường cong huấn luyện).

## Chạy trên Kaggle

Dùng file `hand_gesture_kaggle.ipynb` (đã chứa toàn bộ code):
File → Import Notebook → Add Input `hagrid-sample-30k-384p` → bật GPU → Run All.

## Chạy từng bước (máy có GPU)

```
python -m src.data.prepare_hagrid --ann_dir <HaGRID>/ann_train_val --img_dir <HaGRID> --out_dir data/processed
python -m src.data.split --data_dir data/processed
python -m src.train_classical --model svm rf knn
python -m src.train_dl --model cnn --epochs 40
python -m src.train_dl --model mobilenet_v2 --strategy finetune
python -m src.compare_results
```
