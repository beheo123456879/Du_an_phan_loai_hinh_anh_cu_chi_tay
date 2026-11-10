"""Độ đo đánh giá + vẽ biểu đồ. Có thể chạy riêng để đánh giá lại một checkpoint:

    python -m src.evaluate --ckpt outputs/mobilenet_v2_finetune/best.pt \
        --data_dir data/processed --custom_dir data/custom
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.utils import get_device, save_json


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y)
    probs = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), probs.argmax(1), probs


def compute_metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "report": classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                        output_dict=True, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, classes, path, title="Confusion matrix", normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1 if normalize else None)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(epochs, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(epochs, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(epochs, [h["val_f1"] for h in history], label="val F1", linestyle="--")
    ax[1].set_title("Accuracy / F1")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_and_save(model, loader, classes, device, out_dir, tag):
    """Dự đoán, tính độ đo, vẽ ma trận nhầm lẫn; trả về dict độ đo."""
    y_true, y_pred, _ = predict(model, loader, device)
    m = compute_metrics(y_true, y_pred, classes)
    plot_confusion_matrix(y_true, y_pred, classes, Path(out_dir) / f"cm_{tag}.png",
                          title=f"{tag}: acc={m['accuracy']:.3f}, F1={m['f1_macro']:.3f}")
    print(f"[{tag}] acc={m['accuracy']:.4f}  P={m['precision_macro']:.4f}  "
          f"R={m['recall_macro']:.4f}  F1={m['f1_macro']:.4f}")
    return m


def main():
    from src.data.dataset import make_custom_loader, make_loaders
    from src.models import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--custom_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    out_dir = Path(args.ckpt).parent
    results = {}
    loaders = make_loaders(args.data_dir, args.batch_size, args.num_workers, img_size=ckpt["img_size"])
    results["test"] = evaluate_and_save(model, loaders["test"], ckpt["classes"], device, out_dir, "test")
    if args.custom_dir:
        cl = make_custom_loader(args.custom_dir, args.batch_size, args.num_workers, ckpt["img_size"])
        results["custom"] = evaluate_and_save(model, cl, ckpt["classes"], device, out_dir, "custom")
    save_json(results, out_dir / "eval.json")


if __name__ == "__main__":
    main()
