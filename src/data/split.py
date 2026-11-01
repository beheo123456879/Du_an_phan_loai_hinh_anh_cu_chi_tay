"""Chia train/val/test = 70/15/15, phân tầng theo lớp, KHÔNG để cùng một người nằm ở 2 tập.

    python -m src.data.split --data_dir /kaggle/working/data/processed
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED


def group_stratified_split(df, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    df = df.reset_index(drop=True).copy()

    # Bước 1: tách test (~1/7 ≈ 15%)
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_ratio), shuffle=True, random_state=seed)
    trval_idx, test_idx = next(sgkf.split(df, df["label"], df["user_id"]))
    df.loc[test_idx, "split"] = "test"

    # Bước 2: tách val từ phần còn lại (~15% tổng)
    trval = df.iloc[trval_idx]
    sgkf = StratifiedGroupKFold(n_splits=round((1 - test_ratio) / val_ratio), shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(trval, trval["label"], trval["user_id"]))
    df.loc[trval.index[tr_idx], "split"] = "train"
    df.loc[trval.index[va_idx], "split"] = "val"
    return df


def check_no_user_leak(df):
    users = {s: set(df.loc[df.split == s, "user_id"]) for s in ["train", "val", "test"]}
    assert not users["train"] & users["test"], "Rò rỉ người dùng giữa train và test!"
    assert not users["train"] & users["val"], "Rò rỉ người dùng giữa train và val!"
    assert not users["val"] & users["test"], "Rò rỉ người dùng giữa val và test!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = group_stratified_split(pd.read_csv(data_dir / "metadata.csv"))
    check_no_user_leak(df)
    df.to_csv(data_dir / "splits.csv", index=False)

    print(pd.crosstab(df["label"], df["split"], margins=True))
    print((df["split"].value_counts(normalize=True) * 100).round(1))


if __name__ == "__main__":
    main()
