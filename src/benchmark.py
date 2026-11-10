"""Số tham số, kích thước mô hình, thời gian suy luận trên CPU (batch = 1).

    python -m src.benchmark --ckpt outputs/*/best.pt
"""
import argparse
import copy
import io
import time

import numpy as np
import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def cpu_latency_ms(model, img_size=224, n_warmup=10, n_runs=100, threads=None):
    if threads:
        torch.set_num_threads(threads)
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(n_warmup):
        model(x)
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "threads": torch.get_num_threads()}


def benchmark_model(model, img_size=224, threads=None):
    return {"params": count_params(model), "size_mb": model_size_mb(model),
            "latency": cpu_latency_ms(model, img_size, threads=threads)}


def main():
    from src.models import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    print(f"{'model':<35}{'params(M)':>10}{'size(MB)':>10}{'CPU ms':>10}")
    for path in args.ckpt:
        model, ckpt = load_checkpoint(path, "cpu")
        b = benchmark_model(model, ckpt["img_size"], args.threads)
        print(f"{path:<35}{b['params'] / 1e6:>10.2f}{b['size_mb']:>10.1f}{b['latency']['mean_ms']:>10.1f}")


if __name__ == "__main__":
    main()
