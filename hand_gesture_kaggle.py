"""End-to-end Kaggle workflow for the HaGRID hand gesture project.

Example:
    python hand_gesture_kaggle.py
    python hand_gesture_kaggle.py --max-per-class 500 --epochs 5
    python hand_gesture_kaggle.py --run-classical
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("/kaggle/input/hagrid-sample-30k-384p/ann_train_val"),
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("/kaggle/input/hagrid-sample-30k-384p/hagrid_30k"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/outputs"))
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet_v2", "efficientnet_b0", "resnet50"],
        default="mobilenet_v2",
    )
    parser.add_argument(
        "--strategy",
        choices=["scratch", "feature_extract", "finetune"],
        default="finetune",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-classical", action="store_true", help="Also train HOG + SVM/RF/KNN.")
    return parser.parse_args()


def run(command, project_dir):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)


def main():
    args = parse_args()
    project_dir = args.project_dir.resolve()
    ann_dir = args.ann_dir.resolve()
    img_dir = args.img_dir.resolve()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not (project_dir / "src" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Project source not found under {project_dir}. "
            "Set --project-dir to the project root containing src/."
        )
    for name, path in (("--ann-dir", ann_dir), ("--img-dir", img_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.max_per_class < 1:
        raise ValueError("--max-per-class must be greater than zero.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--epochs and --batch-size must be positive; --num-workers cannot be negative.")
    if args.model == "cnn" and args.strategy != "scratch":
        raise ValueError("The custom CNN only supports --strategy scratch.")
    if args.model != "cnn" and args.strategy == "scratch":
        raise ValueError("Transfer-learning models require --strategy finetune or feature_extract.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install project requirements in the Kaggle environment.") from exc
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required. Install project requirements in the Kaggle environment.") from exc

    sys.path.insert(0, str(project_dir))
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: running on CPU; training may be slow.")

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "src.data.prepare_hagrid",
            "--ann_dir",
            ann_dir,
            "--img_dir",
            img_dir,
            "--out_dir",
            data_dir,
            "--max_per_class",
            str(args.max_per_class),
        ],
        project_dir,
    )
    run(
        [sys.executable, "-m", "src.data.split", "--data_dir", data_dir],
        project_dir,
    )

    splits = pd.read_csv(data_dir / "splits.csv")
    print("\nImages per split:")
    print(splits["split"].value_counts())
    print("\nImages per class and split:")
    print(pd.crosstab(splits["label"], splits["split"], margins=True))

    users_by_split = {
        split: set(splits.loc[splits["split"] == split, "user_id"])
        for split in ("train", "val", "test")
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = users_by_split[first] & users_by_split[second]
        if overlap:
            raise ValueError(f"User leakage detected between {first} and {second}: {len(overlap)} users.")
    print("Verified: no user_id overlap between train, validation, and test.")

    experiment_name = f"{args.model}_{args.strategy}"
    run(
        [
            sys.executable,
            "-m",
            "src.train_dl",
            "--model",
            args.model,
            "--strategy",
            args.strategy,
            "--data_dir",
            data_dir,
            "--out_dir",
            out_dir,
            "--exp_name",
            experiment_name,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ],
        project_dir,
    )

    experiment_dir = out_dir / experiment_name
    checkpoint = experiment_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training finished but checkpoint was not created: {checkpoint}")
    run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--ckpt",
            checkpoint,
            "--data_dir",
            data_dir,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ],
        project_dir,
    )

    if args.run_classical:
        run(
            [
                sys.executable,
                "-m",
                "src.train_classical",
                "--model",
                "svm",
                "rf",
                "knn",
                "--data_dir",
                data_dir,
                "--out_dir",
                out_dir,
            ],
            project_dir,
        )

    run(
        [sys.executable, "-m", "src.compare_results", "--out_dir", out_dir],
        project_dir,
    )
    print(f"\nDone. Results are saved under {out_dir}.")


if __name__ == "__main__":
    main()
