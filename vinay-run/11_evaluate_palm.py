"""
Step 11 — External Validation on PALM Dataset (Inference-Only)

Evaluates the leakage-controlled model (run_C_clean) on the PALM fundus dataset
as a cross-dataset generalization check. No retraining or fine-tuning is performed.

PALM labels (from filename prefix in Training/Classification/):
  P* = Pathologic Myopia  → mapped to POSITIVE (model's "Myopia")
  H* = High Myopia        → mapped to POSITIVE (myopically affected, not pathologic yet)
  N* = Normal / Non-PM    → mapped to NEGATIVE (model's "Normal")

Label mapping options:
  --mapping conservative  P=Myopia, H+N=Normal  (strictest PM vs non-PM split)
  --mapping inclusive     P+H=Myopia, N=Normal   (myopia-spectrum vs normal, closest to
                                                  your Kaggle task definition)

Both are reported. The paper should present both and discuss the difference.

Why only Training/Classification/ (400 images)?
  - PALM Validation split (V*.jpg, 400 images) has no classification labels in this
    download. The Training split has labels embedded in filenames.
  - This is inference-only: the model has never seen PALM images.

Usage:
    python 11_evaluate_palm.py \\
        --run-dir runs/classify/run_C_clean \\
        --palm-dir PALM/Training/Classification

    python 11_evaluate_palm.py \\
        --run-dir runs/classify/run_C_clean \\
        --palm-dir PALM/Training/Classification \\
        --mapping conservative
"""

import argparse
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def palm_label(filename: str, mapping: str) -> int | None:
    """
    Return 1 (Myopia / positive) or 0 (Normal / negative) from PALM filename prefix.

    conservative: P=1, H=0, N=0  (strict Pathologic Myopia vs everything else)
    inclusive:    P=1, H=1, N=0  (any myopia vs normal only)

    Returns None if the prefix is unrecognised.
    """
    prefix = filename[0].upper()
    if prefix == "P":
        return 1
    if prefix == "N":
        return 0
    if prefix == "H":
        return 1 if mapping == "inclusive" else 0
    return None


def collect_palm_predictions(model, palm_dir: Path, mapping: str):
    images = sorted(
        p for p in palm_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Found {len(images)} images in {palm_dir}")

    # Figure out which index the model uses for "Myopia"
    # Run a dummy inference to get class list
    dummy = model(str(images[0]), verbose=False)[0]
    model_classes = dummy.names        # dict: {0: 'Myopia', 1: 'Normal'} or similar
    myopia_idx = next(
        (k for k, v in model_classes.items() if v.lower() == "myopia"), None
    )
    if myopia_idx is None:
        raise ValueError(
            f"Could not find 'Myopia' in model class list: {model_classes}"
        )
    print(f"Model classes: {model_classes}  (Myopia index: {myopia_idx})")

    y_true, y_pred, y_prob = [], [], []
    skipped = []

    for img_path in images:
        true_label = palm_label(img_path.name, mapping)
        if true_label is None:
            skipped.append(img_path.name)
            continue

        result = model(str(img_path), verbose=False)[0]
        probs = result.probs.data.cpu().numpy()
        pred_idx = int(np.argmax(probs))
        myopia_prob = float(probs[myopia_idx])

        y_true.append(true_label)
        y_pred.append(1 if pred_idx == myopia_idx else 0)
        y_prob.append(myopia_prob)

    if skipped:
        print(f"Skipped {len(skipped)} images with unrecognised prefix: {skipped[:5]}")

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
        model_classes,
        myopia_idx,
    )


def run_evaluation(model, palm_dir: Path, mapping: str, fig_dir: Path, run_name: str):
    label_desc = (
        "P=Myopia, H=Normal, N=Normal"
        if mapping == "conservative"
        else "P=Myopia, H=Myopia, N=Normal"
    )
    print(f"\n{'='*60}")
    print(f"  PALM External Validation  [{mapping}]")
    print(f"  Label mapping: {label_desc}")
    print(f"{'='*60}")

    y_true, y_pred, y_prob, model_classes, myopia_idx = collect_palm_predictions(
        model, palm_dir, mapping
    )

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    print(f"\n  PALM samples used: {len(y_true)}  (positive={n_pos}, negative={n_neg})")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)

    print(f"\n  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}  ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f}  ({rec*100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"{'='*60}\n")

    print(classification_report(y_true, y_pred, target_names=["Non-PM", "PM"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix [rows=True, cols=Predicted, labels=[Non-PM, PM]]:")
    print(cm)
    TN, FP, FN, TP = cm.ravel()
    print(f"  TP (PM correctly identified): {TP}")
    print(f"  TN (Non-PM correctly identified): {TN}")
    print(f"  FP (Non-PM flagged as PM): {FP}")
    print(f"  FN (PM missed): {FN}")

    # Figures
    suffix = f"_{mapping}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_cm(cm, fig_dir, f"{run_name}_palm{suffix}")
    _plot_roc(y_true, y_prob, auc, fig_dir, f"{run_name}_palm{suffix}")
    _plot_pr(y_true, y_prob, ap, fig_dir, f"{run_name}_palm{suffix}")

    return {
        "mapping": mapping,
        "n_total": len(y_true),
        "n_pm": n_pos,
        "n_nonpm": n_neg,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
    }


def _plot_cm(cm, fig_dir, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-PM", "PM"]).plot(
        ax=axes[0], cmap="Purples", colorbar=False, values_format="d"
    )
    axes[0].set_title(f"PALM External — {label}")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=["Non-PM", "PM"]).plot(
        ax=axes[1], cmap="Purples", colorbar=False, values_format=".2f"
    )
    axes[1].set_title(f"Normalized — {label}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"confusion_matrix.png", dpi=150)
    plt.close()


def _plot_roc(y_true, y_prob, auc_val, fig_dir, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {label}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"roc_curve.png", dpi=150)
    plt.close()


def _plot_pr(y_true, y_prob, ap_val, fig_dir, label):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve — {label}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"pr_curve.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Inference-only external validation on PALM dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir", type=Path, default=Path("runs/classify/run_C_clean"),
        help="Training run directory (default: run_C_clean)",
    )
    ap.add_argument(
        "--palm-dir", type=Path, default=Path("PALM/Training/Classification"),
        help="PALM labeled image folder (default: PALM/Training/Classification)",
    )
    ap.add_argument(
        "--mapping", choices=["conservative", "inclusive", "both"],
        default="both",
        help=(
            "conservative: P=Myopia only; "
            "inclusive: P+H=Myopia; "
            "both: run and report both (default)"
        ),
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Root for figures/ and results/ output",
    )
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    weights = run_dir / "weights" / "best.pt"
    run_name = run_dir.name

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not args.palm_dir.exists():
        raise FileNotFoundError(f"PALM directory not found: {args.palm_dir}")

    print(f"\nLoading model: {weights}")
    model = YOLO(str(weights))

    mappings = (
        ["conservative", "inclusive"] if args.mapping == "both" else [args.mapping]
    )

    all_results = []
    for m in mappings:
        fig_dir = args.output_dir / "figures" / f"{run_name}_palm_{m}"
        result = run_evaluation(model, args.palm_dir, m, fig_dir, run_name)
        all_results.append(result)
        print(f"\nFigures saved to: {fig_dir}/")

    # Save JSON summary
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_out = results_dir / "palm_external_validation.json"
    json_out.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results saved to: {json_out}")

    # Append to external CSV
    csv_out = results_dir / "results_summary_external.csv"
    write_header = not csv_out.exists()
    with open(csv_out, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run", "dataset", "mapping", "n_total", "n_pm", "n_nonpm",
                "accuracy", "precision", "recall", "f1", "roc_auc",
                "avg_precision", "TP", "TN", "FP", "FN",
            ])
        for r in all_results:
            writer.writerow([
                run_name, "PALM", r["mapping"],
                r["n_total"], r["n_pm"], r["n_nonpm"],
                r["accuracy"], r["precision"], r["recall"], r["f1"],
                r["roc_auc"], r["avg_precision"],
                r["TP"], r["TN"], r["FP"], r["FN"],
            ])
    print(f"Results appended to: {csv_out}")


if __name__ == "__main__":
    main()
