"""
Step 1 — Random Balanced Sampling + Train/Val/Test Split

Creates a YOLO-compatible folder structure from the full Kaggle IMAGES/ folder.
Uses a fixed random seed for reproducibility.

Usage (no normalization — matches paper baseline):
    python 1_sample_data.py --images-dir ../IMAGES --output-dir data_10k --n-per-class 5000

Usage (CLAHE normalization — removes brightness shortcut):
    python 1_sample_data.py --images-dir ../IMAGES --output-dir data_10k_clahe --n-per-class 5000 --clahe

For Google Colab:
    python 1_sample_data.py --images-dir /content/.../IMAGES --output-dir /content/data_10k --n-per-class 5000
    python 1_sample_data.py --images-dir /content/.../IMAGES --output-dir /content/data_10k_clahe --n-per-class 5000 --clahe

CLAHE (Contrast Limited Adaptive Histogram Equalization):
    Equalizes brightness locally per image, removing the systematic class-level
    brightness difference (~102 Myopia vs ~64 Normal mean pixel value).
    If accuracy drops significantly with --clahe, it confirms the model was
    exploiting brightness rather than anatomical retinal features.

Requirements for --clahe:
    pip install opencv-python-headless Pillow
"""

import argparse
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}

SOURCE_CLASS_MAP = {
    "Myopia_images": "Myopia",
    "Normal_images": "Normal",
}


def apply_clahe(src_path: Path, dest_path: Path) -> None:
    """Apply CLAHE to a single image and save to dest_path."""
    import cv2
    import numpy as np

    img = cv2.imread(str(src_path))
    if img is None:
        # Fallback: copy as-is if OpenCV can't read it
        shutil.copy2(src_path, dest_path)
        return

    # Convert BGR → LAB, apply CLAHE to L channel only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Save with same extension as source
    cv2.imwrite(str(dest_path), img_eq)


def copy_or_clahe(src: Path, dest: Path, use_clahe: bool) -> None:
    if use_clahe:
        apply_clahe(src, dest)
    else:
        shutil.copy2(src, dest)


def sample_and_split(
    images_dir: Path,
    output_dir: Path,
    n_per_class: int,
    use_clahe: bool = False,
) -> None:
    if use_clahe:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError(
                "opencv-python-headless is required for --clahe.\n"
                "Install it with: pip install opencv-python-headless"
            )

    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    split_counts = {}

    for src_folder_name, class_name in SOURCE_CLASS_MAP.items():
        src_folder = images_dir / src_folder_name
        if not src_folder.exists():
            raise FileNotFoundError(f"Source folder not found: {src_folder}")

        all_files = [
            p for p in src_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]

        if len(all_files) < n_per_class:
            raise ValueError(
                f"{src_folder_name} has only {len(all_files)} images, "
                f"but {n_per_class} requested."
            )

        # Random sample — key improvement over taking first N
        sampled = random.sample(all_files, n_per_class)

        n_train = int(n_per_class * TRAIN_RATIO)
        n_val   = int(n_per_class * VAL_RATIO)

        splits = {
            "train":      sampled[:n_train],
            "validation": sampled[n_train : n_train + n_val],
            "test":       sampled[n_train + n_val :],
        }

        split_counts[class_name] = {k: len(v) for k, v in splits.items()}

        for split_name, files in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for i, f in enumerate(files):
                copy_or_clahe(f, dest_dir / f.name, use_clahe)
                total_copied += 1
                if use_clahe and (i + 1) % 500 == 0:
                    print(f"  {class_name}/{split_name}: {i+1}/{len(files)} processed...")

    mode_str = "CLAHE-normalized" if use_clahe else "raw (no normalization)"
    print(f"\nDataset created at: {output_dir}")
    print(f"Mode: {mode_str}")
    print(f"Seed: {SEED}  |  Images per class: {n_per_class}")
    print(f"Split ratios: train={TRAIN_RATIO} / val={VAL_RATIO} / test={TEST_RATIO}\n")
    print(f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 44)
    for cls, counts in split_counts.items():
        total = sum(counts.values())
        print(f"{cls:<12} {counts['train']:>8} {counts['validation']:>8} {counts['test']:>8} {total:>8}")
    print("-" * 44)
    grand_train = sum(v["train"] for v in split_counts.values())
    grand_val   = sum(v["validation"] for v in split_counts.values())
    grand_test  = sum(v["test"] for v in split_counts.values())
    print(f"{'TOTAL':<12} {grand_train:>8} {grand_val:>8} {grand_test:>8} {total_copied:>8}")
    print(f"\nDone. {total_copied} images {'CLAHE-processed' if use_clahe else 'copied'}.")

    if use_clahe:
        print("\nNote: CLAHE equalizes local contrast per image, removing")
        print("class-level brightness differences. If accuracy drops vs the")
        print("raw split, the model was relying on brightness as a shortcut.")


def main():
    parser = argparse.ArgumentParser(
        description="Random balanced sampling + YOLO split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../IMAGES"),
        help="Path to the Kaggle IMAGES/ folder (contains Myopia_images/ and Normal_images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_10k"),
        help="Where to write the split dataset",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=5000,
        help="Number of images to sample per class (default: 5000 → 10k total)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        default=False,
        help=(
            "Apply CLAHE brightness normalization to every image before saving. "
            "Removes class-level brightness bias. Requires opencv-python-headless."
        ),
    )
    args = parser.parse_args()

    sample_and_split(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        use_clahe=args.clahe,
    )


if __name__ == "__main__":
    main()
