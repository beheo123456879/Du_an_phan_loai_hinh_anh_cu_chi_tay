"""Huấn luyện CNN tự thiết kế / MobileNetV2 / EfficientNet-B0 / ResNet50.

Chiến lược (--strategy):
    scratch          : huấn luyện toàn bộ từ đầu (mặc định cho cnn)
    feature_extract  : đóng băng mạng nền, chỉ học lớp phân loại
    finetune         : học lớp phân loại --head_epochs epoch, sau đó mở băng các khối cuối (mặc định cho transfer)

Ví dụ:
    python -m src.train_dl --model cnn --epochs 40 --lr 1e-3
    python -m src.train_dl --model mobilenet_v2 --strategy finetune
    python -m src.train_dl --model mobilenet_v2 --strategy finetune --no_aug --exp_name mbv2_noaug   # ablation
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.benchmark import benchmark_model
from src.config import CLASSES, DL_MODELS, IMG_SIZE, NUM_CLASSES, SEED
from src.data.dataset import make_custom_loader, make_loaders
from src.evaluate import evaluate_and_save, plot_history
from src.models import build_model, load_checkpoint
from src.models.transfer import freeze_backbone, freeze_bn_stats, unfreeze_last_blocks
from src.utils import get_device, save_json, seed_everything


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=DL_MODELS)
    ap.add_argument("--strategy", choices=["scratch", "feature_extract", "finetune"], default=None)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--custom_dir", default=None, help="tập tự thu thập để đánh giá tổng quát hóa")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--head_epochs", type=int, default=5, help="số epoch chỉ học head trước khi fine-tune")
    ap.add_argument("--unfreeze_blocks", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="learning rate giai đoạn fine-tune")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    ap.add_argument("--patience", type=int, default=7, help="early stopping")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--no_aug", action="store_true", help="tắt tăng cường dữ liệu (ablation)")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.strategy is None:
        args.strategy = "scratch" if args.model == "cnn" else "finetune"
    if args.model == "cnn" and args.strategy != "scratch":
        ap.error("CNN tự thiết kế chỉ dùng --strategy scratch")
    if args.exp_name is None:
        args.exp_name = f"{args.model}_{args.strategy}" + ("_noaug" if args.no_aug else "")
    return args


def make_optimizer(model, args, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    return opt_cls(params, lr=lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, args, n_epochs):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, n_epochs))
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    if train:
        freeze_bn_stats(model)
    total_loss, preds, targets = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return (total_loss / len(loader.dataset), accuracy_score(targets, preds),
            f1_score(targets, preds, average="macro"))


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    exp_dir = Path(args.out_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(exp_dir / "tb")
    print(f"Thực nghiệm: {args.exp_name} | thiết bị: {device}")

    loaders = make_loaders(args.data_dir, args.batch_size, args.num_workers, not args.no_aug, args.img_size)
    print({k: len(v.dataset) for k, v in loaders.items()})

    model = build_model(args.model, NUM_CLASSES, pretrained=args.strategy != "scratch").to(device)
    if args.strategy in ("feature_extract", "finetune"):
        freeze_backbone(model, args.model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    first_stage_epochs = args.head_epochs if args.strategy == "finetune" else args.epochs
    optimizer = make_optimizer(model, args, args.lr)
    scheduler = make_scheduler(optimizer, args, first_stage_epochs)

    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    ckpt_path = exp_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        in_head_stage = args.strategy == "finetune" and epoch <= args.head_epochs
        if args.strategy == "finetune" and epoch == args.head_epochs + 1:
            unfreeze_last_blocks(model, args.model, args.unfreeze_blocks)
            optimizer = make_optimizer(model, args, args.ft_lr)
            scheduler = make_scheduler(optimizer, args, args.epochs - args.head_epochs)
            bad_epochs = 0
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"==> Bắt đầu fine-tune, số tham số huấn luyện: {n_train / 1e6:.2f}M")

        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(va_f1)
        else:
            scheduler.step()

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1, lr=lr))
        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch)
        writer.add_scalar("val_f1", va_f1, epoch)
        writer.add_scalar("lr", lr, epoch)
        print(f"Epoch {epoch:02d} | lr {lr:.1e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = va_f1, epoch, 0
            torch.save({"model_name": args.model, "state_dict": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "epoch": epoch, "val_f1": va_f1, "args": vars(args)}, ckpt_path)
        elif not in_head_stage:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping tại epoch {epoch}")
                break
    writer.close()
    plot_history(history, exp_dir / "history.png")

    # ---- Đánh giá mô hình tốt nhất ----
    model, _ = load_checkpoint(ckpt_path, device)
    results = {"exp_name": args.exp_name, "model": args.model, "strategy": args.strategy,
               "augment": not args.no_aug, "best_epoch": best_epoch, "val_f1": best_f1, "history": history}
    results["test"] = evaluate_and_save(model, loaders["test"], CLASSES, device, exp_dir, "test")
    if args.custom_dir:
        cl = make_custom_loader(args.custom_dir, args.batch_size, args.num_workers, args.img_size)
        results["custom"] = evaluate_and_save(model, cl, CLASSES, device, exp_dir, "custom")

    b = benchmark_model(model, args.img_size)
    results.update(params=b["params"], size_mb=b["size_mb"], latency_ms=b["latency"]["mean_ms"])
    print(f"Tham số: {b['params'] / 1e6:.2f}M | {b['size_mb']:.1f}MB | CPU {b['latency']['mean_ms']:.1f} ms/ảnh")
    save_json(results, exp_dir / "results.json")


if __name__ == "__main__":
    main()
