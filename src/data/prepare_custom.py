"""Chuẩn bị tập dữ liệu tự thu thập (chỉ dùng để KIỂM THỬ ngoài phân phối).

Đầu vào: data/custom_raw/<class>/*.jpg (ảnh chụp từ webcam/điện thoại)
Đầu ra : data/custom/<class>/*.jpg + data/custom/metadata.csv

    # Ảnh đã cắt sẵn vùng tay:
    python -m src.data.prepare_custom --in_dir data/custom_raw --out_dir data/custom
    # Ảnh nguyên khung -> dùng MediaPipe cắt tay (giống lúc chạy ứng dụng thật):
    python -m src.data.prepare_custom --in_dir data/custom_raw --out_dir data/custom --use_mediapipe
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import CLASSES, IMG_SIZE

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/custom_raw")
    ap.add_argument("--out_dir", default="data/custom")
    ap.add_argument("--use_mediapipe", action="store_true")
    ap.add_argument("--margin", type=float, default=0.25)
    args = ap.parse_args()

    detector = None
    if args.use_mediapipe:
        from src.hand_detector import HandDetector  # import muộn để Kaggle không cần cài mediapipe
        detector = HandDetector(static=True, margin=args.margin)

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    rows, skipped = [], 0
    for cls in CLASSES:
        files = [p for p in (in_dir / cls).glob("*") if p.suffix.lower() in IMG_EXTS]
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
        for p in tqdm(files, desc=cls):
            img = Image.open(p).convert("RGB")
            if detector is not None:
                boxes = detector.detect(np.asarray(img))
                if not boxes:
                    skipped += 1
                    continue
                img = img.crop(boxes[0])
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            rel = f"{cls}/{p.stem}.jpg"
            img.save(out_dir / rel, quality=95)
            rows.append({"path": rel, "label": cls, "user_id": "custom", "source": "custom", "split": "custom"})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(df["label"].value_counts())
    print(f"Tổng {len(df)} ảnh, bỏ qua {skipped} ảnh không phát hiện được tay")


if __name__ == "__main__":
    main()
