"""Cắt vùng bàn tay từ HaGRID theo bounding box và lưu thành data/processed/<class>/*.jpg.

Định dạng annotation HaGRID (mỗi lớp 1 file <class>.json):
    { "<image_id>": { "bboxes": [[x, y, w, h], ...],   # tọa độ chuẩn hóa [0, 1], (x, y) là góc trên-trái
                      "labels": ["like", "no_gesture"],
                      "user_id": "..." }, ... }

Ví dụ trên Kaggle (dataset "hagrid-sample-30k-384p"):
    python -m src.data.prepare_hagrid \
        --ann_dir /kaggle/input/hagrid-sample-30k-384p/ann_train_val \
        --img_dir /kaggle/input/hagrid-sample-30k-384p/hagrid_30k \
        --out_dir /kaggle/working/data/processed --max_per_class 2000
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import CLASSES, IMG_SIZE, SEED
from src.data.crop import square_box

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def build_image_index(img_root):
    """Map image_id -> đường dẫn file (quét đệ quy, vì mỗi bản HaGRID xếp thư mục hơi khác nhau)."""
    return {p.stem: p for p in Path(img_root).rglob("*") if p.suffix.lower() in IMG_EXTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--max_per_class", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.15, help="mở rộng biên mỗi phía (10-20%%)")
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print("Đang lập chỉ mục ảnh...")
    index = build_image_index(args.img_dir)
    print(f"Tìm thấy {len(index)} ảnh")

    rows = []
    for cls in args.classes:
        ann_path = Path(args.ann_dir) / f"{cls}.json"
        if not ann_path.exists():
            print(f"[CẢNH BÁO] Không có {ann_path}, bỏ qua lớp {cls}")
            continue
        with open(ann_path, encoding="utf-8") as f:
            anns = json.load(f)

        ids = list(anns.keys())
        random.Random(SEED).shuffle(ids)
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

        n = 0
        for img_id in tqdm(ids, desc=cls):
            if n >= args.max_per_class:
                break
            if img_id not in index:
                continue
            ann = anns[img_id]
            if cls not in ann["labels"]:
                continue
            bx, by, bw, bh = ann["bboxes"][ann["labels"].index(cls)]

            img = Image.open(index[img_id]).convert("RGB")
            W, H = img.size
            box = square_box(bx * W, by * H, (bx + bw) * W, (by + bh) * H, W, H, args.margin)
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            crop = img.crop(box).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            rel = f"{cls}/{img_id}.jpg"
            crop.save(out_dir / rel, quality=95)
            rows.append({"path": rel, "label": cls, "user_id": ann.get("user_id", img_id), "source": "hagrid"})
            n += 1
        print(f"{cls}: {n} ảnh")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Đã lưu {len(df)} ảnh, {df['user_id'].nunique()} người -> {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
