"""
Step 4 — Threshold Optimization

The paper uses the default 0.5 threshold. Because ROC AUC (0.813) > accuracy (68%),
a better threshold almost certainly exists. This script sweeps thresholds and finds:

  - Best F1 threshold       (best overall balance)
  - Best screening threshold (maximize recall, keep precision >= 0.5)

No retraining needed — runs on the existing model.

Usage:
    python 4_threshold_sweep.py --run-dir runs/classify/run_C --test-dir data_10k/test
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
)
from ultralytics import YOLO

from runs_utils import resolve_run_dir


POSITIVE_CLASS = "Myopia"
MIN_PRECISION_FOR_SCREENING = 0.50  # minimum acceptable precision for screening use


def collect_probs(model, test_dir: Path):
    """Collect ground truth and Myopia class probabilities."""
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    pos_idx = class_names.index(POSITIVE_CLASS)

    y_true, y_probs = [], []

    for class_name in class_names:
        class_dir = test_dir / class_name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        true_bin = 1 if class_name == POSITIVE_CLASS else 0

        for img_path in images:
            result = model(str(img_path), verbose=False)[0]
            probs = result.probs.data.cpu().numpy()
            y_true.append(true_bin)
            y_probs.append(float(probs[pos_idx]))

    return np.array(y_true), np.array(y_probs)


def sweep_thresholds(y_true, y_probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": t, "acc": acc, "prec": prec, "rec": rec, "f1": f1})

    return rows


def find_best_thresholds(rows, y_true, y_probs):
    # Best F1
    best_f1_row = max(rows, key=lambda r: r["f1"])

    # Best recall while maintaining acceptable precision (for screening)
    eligible = [r for r in rows if r["prec"] >= MIN_PRECISION_FOR_SCREENING]
    best_screening_row = max(eligible, key=lambda r: r["rec"]) if eligible else None

    return best_f1_row, best_screening_row


def print_table(rows, best_f1_row, best_screening_row):
    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}  Note")
    print("-" * 65)

    for r in rows:
        t = r["threshold"]
        note = ""
        if abs(t - 0.5) < 0.001:
            note = "<-- paper default"
        if t == best_f1_row["threshold"]:
            note = "<-- best F1"
        if best_screening_row and t == best_screening_row["threshold"] and note == "":
            note = "<-- best screening recall"

        print(
            f"{r['threshold']:>10.2f} {r['acc']:>10.3f} {r['prec']:>10.3f} "
            f"{r['rec']:>10.3f} {r['f1']:>10.3f}  {note}"
        )

    print(f"\n{'Best F1 threshold':30s}: {best_f1_row['threshold']:.2f}")
    print(f"  Accuracy:  {best_f1_row['acc']:.3f}  |  Precision: {best_f1_row['prec']:.3f}")
    print(f"  Recall:    {best_f1_row['rec']:.3f}  |  F1:        {best_f1_row['f1']:.3f}")

    if best_screening_row:
        print(f"\n{'Best screening threshold':30s}: {best_screening_row['threshold']:.2f}")
        print(f"  (Recall maximized while Precision >= {MIN_PRECISION_FOR_SCREENING})")
        print(f"  Accuracy:  {best_screening_row['acc']:.3f}  |  Precision: {best_screening_row['prec']:.3f}")
        print(f"  Recall:    {best_screening_row['rec']:.3f}  |  F1:        {best_screening_row['f1']:.3f}")


def plot_threshold_sweep(rows, fig_dir: Path, run_name: str):
    thresholds = [r["threshold"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, [r["acc"]  for r in rows], label="Accuracy",  lw=2)
    plt.plot(thresholds, [r["prec"] for r in rows], label="Precision", lw=2)
    plt.plot(thresholds, [r["rec"]  for r in rows], label="Recall",    lw=2)
    plt.plot(thresholds, [r["f1"]   for r in rows], label="F1 Score",  lw=2, linestyle="--")

    best_f1 = max(rows, key=lambda r: r["f1"])
    plt.axvline(x=best_f1["threshold"], color="gray", linestyle=":", label=f"Best F1 @ {best_f1['threshold']:.2f}")
    plt.axvline(x=0.5, color="black", linestyle=":", alpha=0.5, label="Default (0.50)")

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Sweep — {run_name}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_sweep.png", dpi=150)
    plt.close()
    print(f"\nThreshold sweep plot saved to: {fig_dir / 'threshold_sweep.png'}")


def main():
    parser = argparse.ArgumentParser(description="Threshold optimization for best model")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Training run directory (e.g. runs/classify/run_C)",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data_10k/test"),
        help="Test split folder",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Root directory for figures/ output",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    weights = run_dir / "weights" / "best.pt"
    run_name = run_dir.name
    fig_dir = args.output_dir / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Collecting probabilities on test set: {args.test_dir}")
    y_true, y_probs = collect_probs(model, args.test_dir)

    roc_auc = roc_auc_score(y_true, y_probs)
    print(f"ROC AUC: {roc_auc:.4f}\n")

    rows = sweep_thresholds(y_true, y_probs)
    best_f1_row, best_screening_row = find_best_thresholds(rows, y_true, y_probs)
    print_table(rows, best_f1_row, best_screening_row)
    plot_threshold_sweep(rows, fig_dir, run_name)


if __name__ == "__main__":
    main()
