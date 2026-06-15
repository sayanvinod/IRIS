"""
Step 2 — Training Script

Trains a YOLOv8 classification model with configurable parameters.
Each call produces one run in runs/classify/<run-name>/.

Usage examples:

  # Run A: Reproduce paper baseline (YOLOv8n, 64px, 20 epochs)
  python 2_train.py --data data_10k --model yolov8n-cls.pt --imgsz 64 --epochs 20 --name run_A

  # Run B: Resolution upgrade (YOLOv8n, 224px, 50 epochs)
  python 2_train.py --data data_10k --model yolov8n-cls.pt --imgsz 224 --epochs 50 --name run_B

  # Run C: Model + resolution upgrade (YOLOv8s, 224px, 50 epochs)
  python 2_train.py --data data_10k --model yolov8s-cls.pt --imgsz 224 --epochs 50 --name run_C

  # Run D: More data (YOLOv8s, 224px, 50 epochs, 20k images)
  python 2_train.py --data data_20k --model yolov8s-cls.pt --imgsz 224 --epochs 50 --name run_D

For Google Colab, prefix paths with /content/drive/MyDrive/... as needed.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO, settings as yolo_settings


AUGMENTATION = {
    # Mild augmentation appropriate for medical fundus images
    # No vertical flip (upside-down fundus is anatomically invalid)
    "fliplr":   0.5,   # horizontal flip (left/right eye symmetry)
    "flipud":   0.0,   # no vertical flip
    "degrees":  5.0,   # small rotation (camera tilt variation)
    "scale":    0.1,   # mild zoom (distance variation)
    "hsv_h":    0.01,  # tiny hue shift
    "hsv_s":    0.3,   # moderate saturation (lighting variation)
    "hsv_v":    0.3,   # brightness jitter — helps reduce brightness shortcut
    "mosaic":   0.0,   # disabled — not meaningful for classification
}


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    data_dir: str,
    model_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    run_name: str,
    patience: int,
):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            f"Run 1_sample_data.py first."
        )

    print(f"\n{'='*60}")
    print(f"  Run:    {run_name}")
    print(f"  Model:  {model_name}")
    print(f"  imgsz:  {imgsz}")
    print(f"  epochs: {epochs}  (patience={patience})")
    print(f"  batch:  {batch}")
    print(f"  data:   {data_path.resolve()}")
    device = resolve_device()
    print(f"  device: {device}")
    print(f"{'='*60}\n")

    # Force Ultralytics to save under cwd/runs so paths are predictable.
    # Use an absolute path and set runs_dir to cwd so project="classify"
    # produces runs/classify/<name>/ without double-nesting.
    yolo_settings.update({"runs_dir": str(Path.cwd())})

    model = YOLO(model_name)

    model.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        pretrained=True,
        optimizer="auto",
        project="classify",   # relative to runs_dir → saves to runs/classify/<name>/
        name=run_name,
        exist_ok=False,
        plots=True,
        seed=42,
        deterministic=True,
        device=device,
        **AUGMENTATION,
    )

    save_dir = Path(model.trainer.save_dir)
    weights_path = save_dir / "weights" / "best.pt"

    # Confirm weights actually landed where expected
    expected = Path("runs") / "classify" / run_name
    if not weights_path.exists():
        print(f"WARNING: weights not found at expected path: {weights_path}")
        print(f"Actual save_dir from trainer: {save_dir.resolve()}")
    else:
        print(f"\nTraining complete.")
        print(f"Run directory:  {save_dir.resolve()}")
        print(f"Best weights:   {weights_path.resolve()}")
        print(f"\nEvaluate with:")
        print(f"  python 3_evaluate.py --run-dir runs/classify/{run_name} --test-dir <data>/test")

    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification model")
    parser.add_argument(
        "--data", type=str, default="data_10k",
        help="Path to split dataset (output of 1_sample_data.py)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n-cls.pt",
        choices=["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"],
        help="YOLOv8 model variant",
    )
    parser.add_argument(
        "--imgsz", type=int, default=224,
        help="Input image size (64, 128, or 224)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs (early stopping via patience)",
    )
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Batch size (reduce to 16 if Colab runs out of memory)",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (epochs without val improvement)",
    )
    parser.add_argument(
        "--name", type=str, default="run_B",
        help="Run name (used as output folder name)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data,
        model_name=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        run_name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
