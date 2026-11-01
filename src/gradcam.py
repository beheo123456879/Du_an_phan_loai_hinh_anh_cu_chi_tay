"""Grad-CAM: kiểm tra mô hình có nhìn vào vùng bàn tay hay không.

    python -m src.gradcam --ckpt outputs/mobilenet_v2_finetune/best.pt --data_dir data/processed --n 16
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        self.handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        """x: tensor (1, 3, H, W) đã chuẩn hóa. Trả về (cam HxW trong [0,1], class_idx, probs)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits, 1)[0]
        return cam.detach().cpu().numpy(), class_idx, probs.detach().cpu().numpy()

    def remove(self):
        self.handle.remove()


def overlay_cam(rgb, cam, alpha=0.45):
    """rgb: uint8 HxWx3 (RGB). Trả về ảnh RGB đã chồng heatmap."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    return np.uint8(alpha * heat + (1 - alpha) * rgb)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.data.dataset import load_split
    from src.data.transforms import get_eval_transforms
    from src.models import get_gradcam_layer, load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--csv", default=None, help="mặc định <data_dir>/splits.csv; dùng data/custom/metadata.csv cho tập tự thu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt, "cpu")
    classes, size = ckpt["classes"], ckpt["img_size"]
    cam_fn = GradCAM(model, get_gradcam_layer(model, ckpt["model_name"]))
    tf = get_eval_transforms(size)

    df = load_split(args.csv or Path(args.data_dir) / "splits.csv", args.split)
    df = df.sample(min(args.n, len(df)), random_state=0)

    cols = 4
    rows = math.ceil(len(df) / cols)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 2.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, r) in enumerate(df.iterrows()):
        img = Image.open(Path(args.data_dir) / r["path"]).convert("RGB").resize((size, size))
        cam, pred, probs = cam_fn(tf(img).unsqueeze(0))
        rgb = np.asarray(img)
        ok = classes[pred] == r["label"]
        a1, a2 = axes[i // cols, (i % cols) * 2], axes[i // cols, (i % cols) * 2 + 1]
        a1.imshow(rgb)
        a1.set_title(f"GT: {r['label']}", fontsize=9)
        a2.imshow(overlay_cam(rgb, cam))
        a2.set_title(f"Pred: {classes[pred]} ({probs[pred]:.2f})", fontsize=9, color="green" if ok else "red")
    plt.tight_layout()
    out = args.out or Path(args.ckpt).parent / f"gradcam_{args.split}.png"
    plt.savefig(out, dpi=130)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
