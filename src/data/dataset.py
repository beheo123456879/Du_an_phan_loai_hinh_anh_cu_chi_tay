from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import CLASS_TO_IDX
from src.data.transforms import get_eval_transforms, get_train_transforms


class GestureDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.root = Path(root)
        self.paths = df["path"].tolist()
        self.labels = [CLASS_TO_IDX[l] for l in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def load_split(csv_path, split):
    df = pd.read_csv(csv_path)
    return df[df["split"] == split].reset_index(drop=True)


def make_loader(df, root, transform, batch_size=64, shuffle=False, num_workers=2):
    return DataLoader(
        GestureDataset(df, root, transform), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def make_loaders(data_dir, batch_size=64, num_workers=2, augment=True, img_size=224):
    """Trả về dict {'train', 'val', 'test'} DataLoader đọc từ <data_dir>/splits.csv."""
    csv = Path(data_dir) / "splits.csv"
    train_tf, eval_tf = get_train_transforms(img_size, augment), get_eval_transforms(img_size)
    return {
        "train": make_loader(load_split(csv, "train"), data_dir, train_tf, batch_size, True, num_workers),
        "val": make_loader(load_split(csv, "val"), data_dir, eval_tf, batch_size, False, num_workers),
        "test": make_loader(load_split(csv, "test"), data_dir, eval_tf, batch_size, False, num_workers),
    }


def make_custom_loader(custom_dir, batch_size=64, num_workers=2, img_size=224):
    df = pd.read_csv(Path(custom_dir) / "metadata.csv")
    return make_loader(df, custom_dir, get_eval_transforms(img_size), batch_size, False, num_workers)
