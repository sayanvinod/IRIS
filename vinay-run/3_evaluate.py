"""
Step 3 — Full Evaluation on Held-Out Test Set

Loads a trained model and evaluates it on the TEST set (not validation).
Outputs: confusion matrix, classification report, ROC curve, PR curve.

Usage:
    python 3_evaluate.py --run-dir runs/run_B --test-dir data_10k/test
    python 3_evaluate.py --run-dir runs/run_C --test-dir data_10k/test
    # Legacy path also works: runs/classify/runs/classify/run_A

Saves figures to:  figures/<run-name>/
Saves CSV to:      results/results_summary.csv  (appended, one row per run)
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from ultralytics import YOLO

from runs_utils import resolve_run_dir


POSITIVE_CLASS = "Myopia"   # class we care about detecting


def collect_predictions(model, test_dir: Path):
    """Run inference on all images in test_dir and return labels + probabilities."""
    class_names = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir()
    ])
    print(f"Classes found: {class_names}")

    if POSITIVE_CLASS not in class_names:
        raise ValueError(
            f"Positive class '{POSITIVE_CLASS}' not found in {test_dir}. "
            f"Found: {class_names}"
        )

    pos_idx = class_names.index(POSITIVE_CLASS)

    true_labels, pred_labels, pred_probs = [], [], []

    for class_name in class_names:
        class_dir = test_dir / class_name
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        true_idx = class_names.index(class_name)
        print(f"  Evaluating {class_name}: {len(images)} images...")

        for img_path in images:
            result = model(str(img_path), verbose=False)[0]
            probs = result.probs.data.cpu().numpy()
            pred_idx = int(np.argmax(probs))

            true_labels.append(true_idx)
            pred_labels.append(pred_idx)
            pred_probs.append(float(probs[pos_idx]))

    return (
        np.array(true_labels),
        np.array(pred_labels),
        np.array(pred_probs),
        class_names,
        pos_idx,
    )


def evaluate(run_dir: Path, test_dir: Path, output_dir: Path, results_csv_name: str = "results_summary.csv"):
    run_dir = resolve_run_dir(run_dir)
    weights = run_dir / "weights" / "best.pt"

    run_name = run_dir.name
    fig_dir = output_dir / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {weights}")
    model = YOLO(str(weights))

    print(f"Evaluating on test set: {test_dir.resolve()}")
    true_labels, pred_labels, pred_probs, class_names, pos_idx = collect_predictions(
        model, test_dir
    )

    # True binary labels for positive class (1 = Myopia)
    y_true_bin = (true_labels == pos_idx).astype(int)
    y_pred_bin = (pred_labels == pos_idx).astype(int)

    acc   = accuracy_score(y_true_bin, y_pred_bin)
    prec  = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec   = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1    = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_true_bin, pred_probs)
    ap    = average_precision_score(y_true_bin, pred_probs)

    print(f"\n{'='*50}")
    print(f"  Results for: {run_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}  ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f}  ({rec*100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"{'='*50}\n")

    print(classification_report(y_true_bin, y_pred_bin, target_names=["Normal", "Myopia"]))

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    print("Confusion Matrix (rows=True, cols=Predicted):")
    print(f"  Labels: [Normal, Myopia]")
    print(cm)
    TN, FP, FN, TP = cm.ravel()
    print(f"\n  TP (Myopia correctly identified): {TP}")
    print(f"  TN (Normal correctly identified): {TN}")
    print(f"  FP (Normal wrongly flagged as Myopia): {FP}")
    print(f"  FN (Myopia missed): {FN}")

    _plot_confusion_matrix(cm, ["Normal", "Myopia"], fig_dir, run_name)
    _plot_roc_curve(y_true_bin, pred_probs, roc_auc, fig_dir, run_name)
    _plot_pr_curve(y_true_bin, pred_probs, ap, fig_dir, run_name)

    # Append to summary CSV
    results_csv = output_dir / "results" / results_csv_name
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    _append_to_csv(results_csv, run_name, acc, prec, rec, f1, roc_auc, ap, TP, TN, FP, FN)

    print(f"\nFigures saved to: {fig_dir}/")
    print(f"Results appended to: {results_csv}")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": roc_auc, "ap": ap}


def _plot_confusion_matrix(cm, class_names, fig_dir, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title(f"Confusion Matrix — {run_name}")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    axes[1].set_title(f"Normalized — {run_name}")

    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_score, roc_auc, fig_dir, run_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {run_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_score, ap, fig_dir, run_name):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {run_name}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", dpi=150)
    plt.close()


def _append_to_csv(csv_path, run_name, acc, prec, rec, f1, auc, ap, TP, TN, FP, FN):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run", "accuracy", "precision", "recall", "f1",
                "roc_auc", "avg_precision", "TP", "TN", "FP", "FN"
            ])
        writer.writerow([
            run_name,
            f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
            f"{auc:.4f}", f"{ap:.4f}",
            TP, TN, FP, FN
        ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to the training run directory (e.g. runs/classify/run_B)",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data_10k/test"),
        help="Path to the test split folder",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Root directory for figures/ and results/ output",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results_summary.csv",
        help="CSV filename under results/ (e.g. results_summary_clean.csv)",
    )
    args = parser.parse_args()

    evaluate(
        run_dir=args.run_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        results_csv_name=args.results_csv,
    )


if __name__ == "__main__":
    main()
