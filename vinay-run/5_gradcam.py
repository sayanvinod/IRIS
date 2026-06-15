"""
Step 5 — Grad-CAM Visualization

Generates Grad-CAM heatmaps showing which retinal regions the model focuses on.
Produces a 2x4 figure grid: 2 Myopia + 2 Normal examples, each with original and heatmap.

This one figure can significantly strengthen the paper by showing the model
attends to anatomical retinal features (optic disc, vessel patterns), not just brightness.

Usage:
    python 5_gradcam.py --run-dir runs/classify/run_C --test-dir data_10k/test

Requirements:
    pip install grad-cam  (pytorch-grad-cam)

NOTE: YOLOv8 does not natively expose Grad-CAM through its API.
This script uses the pytorch-grad-cam library with manual layer targeting.
If you hit issues, the fallback is to use Ultralytics' built-in saliency maps.
"""

import argparse
import random
from pathlib import Path

import matplotlib
# Use the non-interactive Agg backend. The default macOS GUI backend loads a
# second OpenMP runtime which conflicts with PyTorch and aborts (SIGABRT)
# during the Grad-CAM backward pass. Agg also matches our save-to-file usage.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from runs_utils import resolve_run_dir


POSITIVE_CLASS = "Myopia"
SEED = 42
N_SAMPLES_PER_CLASS = 2   # 2 Myopia + 2 Normal = 4 panels total


def load_image(path: Path, imgsz: int = 224):
    """Load and preprocess a single image for display and inference."""
    img = Image.open(path).convert("RGB")
    img_display = img.copy()
    return img_display


def run_ultralytics_gradcam(model_path: Path, image_paths: list, fig_dir: Path, run_name: str):
    """
    Use Ultralytics built-in visualization (most compatible approach).
    Saves individual heatmap images using model.predict with visualize=True.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    fig, axes = plt.subplots(2, len(image_paths), figsize=(4 * len(image_paths), 9))
    fig.suptitle(f"Model Attention — {run_name}", fontsize=14, fontweight="bold")

    for i, (img_path, class_label) in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        # Run prediction
        result = model(str(img_path), verbose=False)[0]
        probs = result.probs.data.cpu().numpy()
        pred_class = result.names[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        # Top row: original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(
            f"True: {class_label}\nPred: {pred_class} ({confidence:.2f})",
            fontsize=9
        )
        axes[0, i].axis("off")

        # Bottom row: attempt Ultralytics feature map visualization
        # Ultralytics saves visualizations to a temp directory with visualize=True
        try:
            result_vis = model(str(img_path), verbose=False, visualize=True)[0]
            axes[1, i].text(
                0.5, 0.5,
                "See visualize/ folder\nfor feature maps",
                ha="center", va="center", transform=axes[1, i].transAxes,
                fontsize=8
            )
        except Exception:
            axes[1, i].text(
                0.5, 0.5, "Visualization\nnot available",
                ha="center", va="center", transform=axes[1, i].transAxes,
                fontsize=8
            )
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = fig_dir / "attention_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


class _YOLOWrapper(torch.nn.Module):
    """Wrap YOLOv8 classify model so forward() returns a plain logits tensor.

    pytorch-grad-cam expects the model to return a tensor, but Ultralytics
    classification models return a tuple (logits, ...) in newer versions.
    This wrapper unwraps the tuple so GradCAM gets what it needs.
    """
    def __init__(self, yolo_torch_model):
        super().__init__()
        self.model = yolo_torch_model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def run_pytorch_gradcam(model_path: Path, image_paths: list, fig_dir: Path, run_name: str):
    """
    Uses pytorch-grad-cam library for proper Grad-CAM heatmaps.
    Install: pip install grad-cam
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam not installed. Run: pip install grad-cam")
        print("Falling back to Ultralytics visualization.")
        run_ultralytics_gradcam(model_path, image_paths, fig_dir, run_name)
        return

    yolo = YOLO(str(model_path))
    class_names = yolo.names
    # Wrap so forward() always returns a plain tensor
    wrapped = _YOLOWrapper(yolo.model)
    # Keep eval mode for BatchNorm/Dropout, but re-enable gradients on all
    # parameters — Ultralytics strips the optimizer and sets requires_grad=False
    # on saved weights, which breaks Grad-CAM.
    wrapped.eval()
    for p in wrapped.parameters():
        p.requires_grad_(True)

    # Target the last conv layer of the backbone (before the classify head).
    # For YOLOv8n/s-cls the backbone is yolo.model.model[:-1]; the classify
    # head is yolo.model.model[-1].  The last conv is model[-2] (C2f block)
    # or model[-3].  We try both and keep the first one that works.
    raw = yolo.model
    target_layer = None
    for idx in [-2, -3, -4]:
        try:
            candidate = raw.model[idx]
            # Make sure it has parameters (not an activation / pool layer)
            if any(True for _ in candidate.parameters()):
                target_layer = [candidate]
                print(f"Grad-CAM target layer: model[{idx}] = {type(candidate).__name__}")
                break
        except Exception:
            continue

    if target_layer is None:
        print("Could not identify target layer. Falling back to Ultralytics visualization.")
        run_ultralytics_gradcam(model_path, image_paths, fig_dir, run_name)
        return

    # Build GradCAM once — reuse across images
    cam_algo = GradCAM(model=wrapped, target_layers=target_layer)

    fig, axes = plt.subplots(2, len(image_paths), figsize=(4 * len(image_paths), 9))
    fig.suptitle(f"Grad-CAM — {run_name}\n(highlighting regions driving predictions)", fontsize=12)

    imgsz = 224

    for i, (img_path, class_label) in enumerate(image_paths):
        img_pil = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
        img_arr = np.array(img_pil).astype(np.float32) / 255.0

        tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0)

        # Predict via the wrapped model (returns softmax probs).
        # IMPORTANT: do NOT use yolo.predict() here — Ultralytics' predict path
        # disables requires_grad on the model params, which breaks the
        # subsequent Grad-CAM backward pass.
        with torch.no_grad():
            probs = wrapped(tensor)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(np.max(probs))

        # Grad-CAM
        try:
            grayscale_cam = cam_algo(input_tensor=tensor, targets=None)
            grayscale_cam = grayscale_cam[0]
            visualization = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
        except Exception as e:
            print(f"Grad-CAM failed for {img_path.name}: {e}")
            visualization = img_arr

        # Original
        axes[0, i].imshow(img_arr)
        axes[0, i].set_title(
            f"True: {class_label}\nPred: {pred_class} ({confidence:.2f})",
            fontsize=9
        )
        axes[0, i].axis("off")

        # Heatmap
        axes[1, i].imshow(visualization)
        axes[1, i].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = fig_dir / "gradcam.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM figure saved: {out_path}")


def pick_samples(test_dir: Path):
    """Pick N_SAMPLES_PER_CLASS images from each class for visualization."""
    random.seed(SEED)
    samples = []

    for class_name in [POSITIVE_CLASS, "Normal"]:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping.")
            continue
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        chosen = random.sample(images, min(N_SAMPLES_PER_CLASS, len(images)))
        samples.extend([(p, class_name) for p in chosen])

    return samples


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for best model")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Training run directory (e.g. runs/classify/run_C)",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data_10k/test"),
        help="Test split folder (to pick sample images from)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Root directory for figures/ output",
    )
    parser.add_argument(
        "--method", choices=["gradcam", "ultralytics"], default="gradcam",
        help="Visualization method (gradcam requires: pip install grad-cam)",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    weights = run_dir / "weights" / "best.pt"
    run_name = run_dir.name
    fig_dir = args.output_dir / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    samples = pick_samples(args.test_dir)
    print(f"Selected {len(samples)} images for visualization:")
    for path, label in samples:
        print(f"  [{label}] {path.name}")

    if args.method == "gradcam":
        run_pytorch_gradcam(weights, samples, fig_dir, run_name)
    else:
        run_ultralytics_gradcam(weights, samples, fig_dir, run_name)


if __name__ == "__main__":
    main()
