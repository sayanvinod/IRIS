"""
Export experiment runs to a local Ultralytics-style runs/ folder.

Usage:
  # Export recovered run_A + run_B metrics (no weights — Colab session lost)
  python 6_export_local_runs.py --recovered

  # After Colab: copy a run folder from Drive onto your Mac, then register it
  python 6_export_local_runs.py --import-run /path/to/run_C

  # Generate confusion matrix PNGs from saved test metrics
  python 6_export_local_runs.py --recovered --plots
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

RUNS_ROOT = Path(__file__).parent / "runs" / "classify"

# Recovered from colab_notebook.ipynb evaluation outputs (June 2026)
RECOVERED_RUNS = {
    "run_A": {
        "config": {
            "task": "classify",
            "model": "yolov8n-cls.pt",
            "data": "data_10k",
            "epochs": 20,
            "patience": 15,
            "batch": 32,
            "imgsz": 64,
            "seed": 42,
            "notes": "Random 10k sample, seed=42. Weights lost when Colab disconnected.",
        },
        "test_metrics": {
            "accuracy": 0.9953,
            "precision": 0.9934,
            "recall": 0.9973,
            "f1": 0.9953,
            "roc_auc": 1.0000,
            "avg_precision": 1.0000,
            "TP": 748, "TN": 745, "FP": 5, "FN": 2,
            "confusion_matrix": [[745, 5], [2, 748]],
            "class_names": ["Normal", "Myopia"],
            "test_set": "data_10k/test (1500 images)",
        },
    },
    "run_B": {
        "config": {
            "task": "classify",
            "model": "yolov8n-cls.pt",
            "data": "data_10k",
            "epochs": 50,
            "patience": 15,
            "batch": 32,
            "imgsz": 224,
            "seed": 42,
            "early_stopping_epoch": 32,
            "best_epoch": 17,
            "notes": "Early stopped epoch 32, best epoch 17. Weights lost when Colab disconnected.",
        },
        "test_metrics": {
            "accuracy": 0.9987,
            "precision": 1.0000,
            "recall": 0.9973,
            "f1": 0.9987,
            "roc_auc": 1.0000,
            "avg_precision": 1.0000,
            "TP": 748, "TN": 750, "FP": 0, "FN": 2,
            "confusion_matrix": [[750, 0], [2, 748]],
            "class_names": ["Normal", "Myopia"],
            "test_set": "data_10k/test (1500 images)",
        },
    },
}


def write_args_yaml(run_dir: Path, config: dict) -> None:
  lines = [f"{k}: {v}" for k, v in config.items()]
  (run_dir / "args.yaml").write_text("\n".join(lines) + "\n")


def write_test_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


def plot_confusion_matrix(run_dir: Path, metrics: dict, run_name: str) -> None:
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["class_names"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title(f"Test Confusion Matrix — {run_name}")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    axes[1].set_title(f"Normalized — {run_name}")

    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def export_recovered(plots: bool = True) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for run_name, data in RECOVERED_RUNS.items():
        run_dir = RUNS_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weights").mkdir(exist_ok=True)

        write_args_yaml(run_dir, data["config"])
        write_test_metrics(run_dir, data["test_metrics"])

        (run_dir / "weights" / "README.txt").write_text(
            "Model weights (best.pt) were not saved — Colab session disconnected.\n"
            "Retrain with:\n"
            f"  python 2_train.py --data data_10k --name {run_name} ...\n"
            "Then copy weights/best.pt here or run with --import-run.\n"
        )

        if plots:
            plot_confusion_matrix(run_dir, data["test_metrics"], run_name)

        print(f"Exported: {run_dir}")


def import_run(source: Path, run_name: str | None = None) -> None:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    name = run_name or source.name
    dest = RUNS_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print(f"Imported {source} → {dest}")


def main():
    parser = argparse.ArgumentParser(description="Export runs to local vinay-run/runs/")
    parser.add_argument("--recovered", action="store_true", help="Export run_A + run_B from recovered metrics")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate confusion matrix PNGs")
    parser.add_argument("--import-run", type=Path, help="Copy a full Colab run folder locally")
    parser.add_argument("--name", type=str, help="Run name when using --import-run")
    args = parser.parse_args()

    if args.recovered:
        export_recovered(plots=args.plots)
    elif args.import_run:
        import_run(args.import_run, args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
