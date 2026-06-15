"""
Modal runner for IRIS run_C_clean — deduplicated clean-split primary model.

Pipeline:
  1. Download Kaggle dataset
  2. Deduplicated cluster-aware sampling (1b_sample_data_clean.py)
  3. Train YOLOv8s, 224px, 50 epochs on data_10k_clean
  4. Evaluate, threshold sweep, Grad-CAM on test set
  5. Persist outputs to Modal volume 'iris-run-c-clean'

━━━ One-time setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. pip install modal
2. modal setup
3. modal secret create kaggle-secret \\
     KAGGLE_USERNAME=<your_username> \\
     KAGGLE_KEY=<your_api_key>

━━━ Run ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  modal run vinay-run/modal_run_c_clean.py

━━━ Download results when done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  modal volume get iris-run-c-clean /runs/classify/run_C_clean ./vinay-run/runs/classify/
  modal volume get iris-run-c-clean /figures/run_C_clean ./vinay-run/figures/
  modal volume get iris-run-c-clean /results/results_summary_clean.csv ./vinay-run/results/
  modal volume get iris-run-c-clean /output/results/leakage_data_10k_clean.txt ./vinay-run/results/leakage_data_10k_clean_modal.txt
  modal volume get iris-run-c-clean /output/results/dedup_manifest.json ./vinay-run/results/dedup_manifest_modal.json
"""

import modal
from modal import FilePatternMatcher
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "ultralytics>=8.0",
        "kagglehub[hf-datasets]",
        "opencv-python-headless",
        "scikit-learn",
        "matplotlib",
        "grad-cam",
        "Pillow",
        "numpy",
    )
    .add_local_dir(
        str(_SCRIPTS_DIR),
        remote_path="/scripts",
        ignore=~FilePatternMatcher("**/*.py"),
    )
)

volume = modal.Volume.from_name("iris-run-c-clean", create_if_missing=True)
VOLUME_PATH = "/output"

app = modal.App("iris-run-c-clean", image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 180,  # 3 hours (dedup ~30 min + train ~60 min + eval)
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    cpu=4,
    memory=32768,
)
def run_c_clean():
    import os
    import shutil
    import subprocess
    from pathlib import Path

    work = Path("/workspace")
    work.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "/scripts", "MPLCONFIGDIR": "/tmp/mplconfig"}
    os.makedirs("/tmp/mplconfig", exist_ok=True)

    def sh(cmd: list[str]) -> None:
        print(f"\n>>> {' '.join(str(c) for c in cmd)}\n{'─'*60}")
        subprocess.run(cmd, check=True, cwd=str(work), env=env)

    # ── 0. Download Kaggle dataset ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 0: Download Kaggle dataset")
    print("=" * 60)
    import kagglehub

    dataset_path = kagglehub.dataset_download("kellysanderson/myopia-image-dataset")
    images_dir = Path(dataset_path) / "IMAGES"
    print(f"\nDataset root:  {dataset_path}")
    print(f"IMAGES folder: {images_dir}")
    assert images_dir.exists(), f"Expected IMAGES/ folder at {images_dir}"

    # ── 1. Deduplicated clean sampling ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Deduplicated clean data split (~30 min for 124k images)")
    print("=" * 60)
    sh([
        "python", "/scripts/1b_sample_data_clean.py",
        "--images-dir", str(images_dir),
        "--output-dir", "data_10k_clean",
        "--n-per-class", "5000",
        "--manifest", "results/dedup_manifest.json",
    ])

    # ── 2. Leakage audit on clean split ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Leakage audit on clean split")
    print("=" * 60)
    sh([
        "python", "/scripts/7_leakage_audit.py",
        "--data-dir", "data_10k_clean",
        "--near-threshold", "5",
        "--report", "results/leakage_data_10k_clean.txt",
    ])

    # ── 3. Train YOLOv8s, 224px, 50 epochs ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Training run_C_clean (~60 min on A10G)")
    print("=" * 60)
    sh([
        "python", "/scripts/2_train.py",
        "--data", "data_10k_clean",
        "--model", "yolov8s-cls.pt",
        "--imgsz", "224",
        "--epochs", "50",
        "--batch", "64",
        "--patience", "15",
        "--name", "run_C_clean",
    ])

    # ── 4. Evaluate on held-out test set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Evaluation on test set")
    print("=" * 60)
    run_hits = list((work / "runs").rglob("run_C_clean/weights/best.pt"))
    if not run_hits:
        raise FileNotFoundError("run_C_clean/weights/best.pt not found after training")
    run_dir = run_hits[0].parent.parent
    print(f"  Found run dir: {run_dir}")

    sh([
        "python", "/scripts/3_evaluate.py",
        "--run-dir", str(run_dir.relative_to(work)),
        "--test-dir", "data_10k_clean/test",
        "--output-dir", ".",
        "--results-csv", "results_summary_clean.csv",
    ])

    # ── 5. Threshold sweep ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Threshold sweep")
    print("=" * 60)
    sh([
        "python", "/scripts/4_threshold_sweep.py",
        "--run-dir", str(run_dir.relative_to(work)),
        "--test-dir", "data_10k_clean/test",
    ])

    # ── 6. Grad-CAM ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: Grad-CAM visualization")
    print("=" * 60)
    sh([
        "python", "/scripts/5_gradcam.py",
        "--run-dir", str(run_dir.relative_to(work)),
        "--test-dir", "data_10k_clean/test",
        "--method", "gradcam",
    ])

    # ── 7. Persist outputs to Modal Volume ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7: Saving to volume 'iris-run-c-clean'")
    print("=" * 60)

    out = Path(VOLUME_PATH)

    def _copy(src: Path, dst: Path) -> None:
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  ✓  {src}  →  {dst}")

    _copy(run_dir, out / "runs" / "classify" / "run_C_clean")
    _copy(work / "figures" / "run_C_clean", out / "figures" / "run_C_clean")
    _copy(work / "results" / "results_summary_clean.csv", out / "results" / "results_summary_clean.csv")
    _copy(work / "results" / "leakage_data_10k_clean.txt", out / "results" / "leakage_data_10k_clean.txt")
    _copy(work / "results" / "dedup_manifest.json", out / "results" / "dedup_manifest.json")

    volume.commit()

    print("\n" + "=" * 60)
    print("  run_C_clean COMPLETE")
    print("=" * 60)
    print("\nDownload to Mac:")
    print("  modal volume get iris-run-c-clean /output/runs/classify/run_C_clean ./vinay-run/runs/classify/run_C_clean")
    print("  modal volume get iris-run-c-clean /output/figures/run_C_clean ./vinay-run/figures/run_C_clean")
    print("  modal volume get iris-run-c-clean /output/results/results_summary_clean.csv ./vinay-run/results/results_summary_clean.csv")


@app.function(volumes={VOLUME_PATH: volume})
def list_outputs():
    import os

    total = 0
    for root, _, files in os.walk(VOLUME_PATH):
        for fname in sorted(files):
            full = os.path.join(root, fname)
            size = os.path.getsize(full)
            total += size
            rel = os.path.relpath(full, VOLUME_PATH)
            print(f"  {rel:<60}  {size/1024:>8.1f} KB")
    print(f"\n  Total: {total/1024/1024:.1f} MB")


@app.local_entrypoint()
def main():
    run_c_clean.remote()
