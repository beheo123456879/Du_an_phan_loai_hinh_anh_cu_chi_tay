"""Gom tất cả outputs/*/results.json thành bảng so sánh + biểu đồ.

    python -m src.compare_results --out_dir outputs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    rows = []
    for f in sorted(out_dir.glob("*/results.json")):
        r = load_json(f)
        rows.append({
            "experiment": r["exp_name"],
            "strategy": r["strategy"],
            "augment": r["augment"],
            "test_acc": r["test"]["accuracy"],
            "test_f1": r["test"]["f1_macro"],
            "test_precision": r["test"]["precision_macro"],
            "test_recall": r["test"]["recall_macro"],
            "custom_acc": r.get("custom", {}).get("accuracy"),
            "custom_f1": r.get("custom", {}).get("f1_macro"),
            "params_M": r["params"] / 1e6 if r.get("params") else None,
            "size_MB": r["size_mb"],
            "cpu_ms": r["latency_ms"],
        })
    if not rows:
        print("Chưa có results.json nào")
        return

    df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    df.to_csv(out_dir / "comparison.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = df.set_index("experiment")
    plot_df[["test_acc", "test_f1", "custom_acc"]].plot.barh(ax=ax[0])
    ax[0].set_xlim(0, 1)
    ax[0].axvline(0.9, color="red", linestyle="--", linewidth=1)
    ax[0].set_title("Accuracy / F1 (đường đỏ = mục tiêu 90%)")
    plot_df["cpu_ms"].plot.barh(ax=ax[1], color="tab:orange")
    ax[1].axvline(50, color="red", linestyle="--", linewidth=1)
    ax[1].set_title("CPU ms/ảnh (mục tiêu < 50)")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=150)
    print(f"Đã lưu {out_dir / 'comparison.csv'} và comparison.png")


if __name__ == "__main__":
    main()
