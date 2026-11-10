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

Workspace có notebook `hand_gesture_kaggle.ipynb` và script `hand_gesture_kaggle.py` để chạy quy trình Kaggle. Có thể dùng notebook tương tác hoặc gọi script từ terminal:

1. Tạo Kaggle Notebook, chọn **Add Input** và thêm dataset `hagrid-sample-30k-384p`.
2. Đưa mã nguồn dự án vào `/kaggle/working/hand_gesture_classification` (ví dụ: thêm project dưới dạng Kaggle Dataset hoặc clone repository nếu bật Internet).
3. Trong **Settings**, chọn **Accelerator → GPU** và bật **Internet** để lần đầu tải trọng số pretrained ImageNet của MobileNetV2.

Để chạy bằng script:

```bash
%cd /kaggle/working/hand_gesture_classification
!python hand_gesture_kaggle.py
```

Có thể tùy chỉnh cấu hình, ví dụ giảm dữ liệu/epoch để chạy thử hoặc bật huấn luyện mô hình cổ điển:

```bash
!python hand_gesture_kaggle.py --max-per-class 500 --epochs 5
!python hand_gesture_kaggle.py --run-classical
```

Script mặc định đọc annotation và ảnh tại `/kaggle/input/hagrid-sample-30k-384p/ann_train_val` và `/kaggle/input/hagrid-sample-30k-384p/hagrid_30k`. Nếu dataset được gắn ở đường dẫn khác, truyền `--ann-dir` và `--img-dir`. Kaggle thường đã cài sẵn PyTorch và các thư viện khoa học dữ liệu. Nếu thiếu module, chỉ cài module được báo thiếu; huấn luyện các mô hình này không cần MediaPipe hay Streamlit.

Chuẩn bị ảnh đã cắt và chia tập. Đường dẫn HaGRID mặc định bên dưới phù hợp với cấu trúc được tài liệu hóa trong `prepare_hagrid.py`; nếu Kaggle gắn dataset ở vị trí khác, hãy thay bằng đường dẫn thực tế bên dưới `/kaggle/input`:

```bash
%cd /kaggle/working/hand_gesture_classification
!python -m src.data.prepare_hagrid \
  --ann_dir /kaggle/input/hagrid-sample-30k-384p/ann_train_val \
  --img_dir /kaggle/input/hagrid-sample-30k-384p/hagrid_30k \
  --out_dir /kaggle/working/data/processed \
  --max_per_class 2000
!python -m src.data.split --data_dir /kaggle/working/data/processed
```

Huấn luyện MobileNetV2 fine-tune:

```bash
%cd /kaggle/working/hand_gesture_classification
!python -m src.train_dl \
  --model mobilenet_v2 \
  --strategy finetune \
  --data_dir /kaggle/working/data/processed \
  --out_dir /kaggle/working/outputs \
  --epochs 30 \
  --batch_size 64
```

Kết quả gồm checkpoint `best.pt`, `results.json`, ma trận nhầm lẫn và biểu đồ huấn luyện tại `/kaggle/working/outputs/mobilenet_v2_finetune/`. Để so sánh các mô hình học máy truyền thống, chạy `python -m src.train_classical --model svm rf knn --data_dir /kaggle/working/data/processed --out_dir /kaggle/working/outputs`. Sau khi chạy xong, chọn **Save Version → Save & Run All** để lưu notebook và kết quả đầu ra.

## Chạy từng bước (máy có GPU)

```
python -m src.data.prepare_hagrid --ann_dir <HaGRID>/ann_train_val --img_dir <HaGRID> --out_dir data/processed
python -m src.data.split --data_dir data/processed
python -m src.train_classical --model svm rf knn
python -m src.train_dl --model cnn --epochs 40
python -m src.train_dl --model mobilenet_v2 --strategy finetune
python -m src.compare_results
```
