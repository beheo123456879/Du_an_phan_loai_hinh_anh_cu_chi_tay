"""Huấn luyện HOG + SVM / Random Forest / KNN với Grid Search + kiểm định chéo theo nhóm người.

    python -m src.train_classical --model svm rf knn --data_dir data/processed --custom_dir data/custom
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.config import CLASS_TO_IDX, CLASSES, CLASSICAL_MODELS, SEED
from src.evaluate import compute_metrics, plot_confusion_matrix
from src.models.classical import PARAM_GRIDS, build_pipeline, extract_hog_features, hog_from_pil
from src.utils import save_json


def load_features(df, root, cache_path, n_jobs):
    if Path(cache_path).exists():
        return np.load(cache_path)
    X = extract_hog_features([Path(root) / p for p in df["path"]], n_jobs)
    np.save(cache_path, X)
    return X


def measure_latency_ms(pipe, paths, n=100):
    """Thời gian 1 ảnh = đọc ảnh + HOG + dự đoán (cùng điều kiện như mô hình DL: batch 1, CPU)."""
    paths = list(paths)[:n]
    times = []
    for p in paths:
        img = Image.open(p)
        img.load()
        t = time.perf_counter()
        pipe.predict(hog_from_pil(img)[None])
        times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=CLASSICAL_MODELS, choices=CLASSICAL_MODELS)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--custom_dir", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--pca", type=int, default=200, help="số thành phần PCA (0 = không dùng)")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "hog_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.data_dir) / "splits.csv")
    # Grid Search dùng CV trên train+val (val đóng vai trò fold), test giữ nguyên để báo cáo
    trval = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    X_tr = load_features(trval, args.data_dir, cache_dir / "trainval.npy", args.n_jobs)
    X_te = load_features(test, args.data_dir, cache_dir / "test.npy", args.n_jobs)
    y_tr = trval["label"].map(CLASS_TO_IDX).values
    y_te = test["label"].map(CLASS_TO_IDX).values
    print(f"HOG: train+val {X_tr.shape}, test {X_te.shape}")

    custom = None
    if args.custom_dir:
        cdf = pd.read_csv(Path(args.custom_dir) / "metadata.csv")
        X_cu = load_features(cdf, args.custom_dir, cache_dir / "custom.npy", args.n_jobs)
        custom = (X_cu, cdf["label"].map(CLASS_TO_IDX).values)

    for name in args.model:
        exp_dir = out_dir / f"hog_{name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== HOG + {name.upper()} =====")

        cv = StratifiedGroupKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
        gs = GridSearchCV(build_pipeline(name, args.pca), PARAM_GRIDS[name], cv=cv,
                          scoring="f1_macro", n_jobs=args.n_jobs, verbose=1, refit=True)
        t0 = time.time()
        gs.fit(X_tr, y_tr, groups=trval["user_id"].values)
        train_time = time.time() - t0
        print(f"Best params: {gs.best_params_} | CV F1 = {gs.best_score_:.4f} | {train_time:.0f}s")

        pipe = gs.best_estimator_
        results = {"exp_name": f"hog_{name}", "model": f"hog_{name}", "strategy": "classical",
                   "augment": False, "best_params": gs.best_params_, "cv_f1": gs.best_score_,
                   "train_time_s": train_time}

        y_pred = pipe.predict(X_te)
        results["test"] = compute_metrics(y_te, y_pred, CLASSES)
        plot_confusion_matrix(y_te, y_pred, CLASSES, exp_dir / "cm_test.png",
                              title=f"HOG+{name.upper()} test: acc={results['test']['accuracy']:.3f}")
        print(f"[test] acc={results['test']['accuracy']:.4f} F1={results['test']['f1_macro']:.4f}")

        if custom is not None:
            y_pred_c = pipe.predict(custom[0])
            results["custom"] = compute_metrics(custom[1], y_pred_c, CLASSES)
            plot_confusion_matrix(custom[1], y_pred_c, CLASSES, exp_dir / "cm_custom.png",
                                  title=f"HOG+{name.upper()} custom")
            print(f"[custom] acc={results['custom']['accuracy']:.4f} F1={results['custom']['f1_macro']:.4f}")

        model_path = exp_dir / "model.joblib"
        joblib.dump(pipe, model_path)
        results["size_mb"] = model_path.stat().st_size / 1e6
        results["params"] = None
        results["latency_ms"] = measure_latency_ms(pipe, (Path(args.data_dir) / p for p in test["path"]))
        print(f"Kích thước {results['size_mb']:.1f}MB | CPU {results['latency_ms']:.1f} ms/ảnh")
        save_json(results, exp_dir / "results.json")


if __name__ == "__main__":
    main()
