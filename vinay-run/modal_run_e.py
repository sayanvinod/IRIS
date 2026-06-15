"""
Modal runner for IRIS run_E — brightness ablation experiment.

Run_E: YOLOv8n, 224px, 50 epochs, 10k images with CLAHE preprocessing.
Hypothesis: if accuracy drops vs run_B (same config, no CLAHE), the model
relied on brightness as a shortcut rather than retinal anatomy.

━━━ One-time setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. pip install modal
2. modal setup                          # authenticate with your account
3. Create a Kaggle secret in Modal:
     modal secret create kaggle-secret \\
       KAGGLE_USERNAME=<your_username> \\
       KAGGLE_KEY=<your_api_key>
   (Find your key at https://www.kaggle.com/settings → API → Create New Token)

━━━ Run ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  modal run vinay-run/modal_run_e.py

━━━ Monitor ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  https://modal.com/apps  →  "iris-run-e"

━━━ Download results when done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Weights + training logs
  modal volume get iris-run-e /output/runs/classify/run_E ./vinay-run/runs/classify/run_E

  # Eval figures (confusion matrix, ROC, PR curve)
  modal volume get iris-run-e /output/figures/run_E ./vinay-run/figures/run_E

  # Results CSV (appended row for run_E)
  modal volume get iris-run-e /output/results/results_summary.csv ./vinay-run/results/results_summary_run_e.csv

  # Or list everything in the volume first:
  modal volume ls iris-run-e /output

━━━ Estimated cost ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A10G GPU: ~$0.63/hr  ×  ~50 min total  ≈  $0.53
  (CLAHE preprocessing ~10 min, training ~30 min, eval ~10 min)
"""

import modal
from modal import FilePatternMatcher
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent

# ── Image — deps + pipeline scripts baked in ──────────────────────────────────
# add_local_dir is the current API (copy_local_dir removed in v0.66.40).
# ~FilePatternMatcher("**/*.py") means "ignore everything except .py files".
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # Full set of system libs required by opencv-python-headless on Debian slim
        "libgl1",           # libGL.so.1
        "libglib2.0-0",     # libgthread-2.0.so.0, libglib-2.0.so.0
        "libsm6",           # libSM.so.6
        "libxext6",         # libXext.so.6
        "libxrender1",      # libXrender.so.1
    )
    .pip_install(
        "ultralytics>=8.0",
        "kagglehub[hf-datasets]",
        "opencv-python-headless",
        "scikit-learn",
        "matplotlib",
    )
    .add_local_dir(
        str(_SCRIPTS_DIR),
        remote_path="/scripts",
        ignore=~FilePatternMatcher("**/*.py"),
    )
)

# ── Persistent volume — survives after the function exits ──────────────────────
volume = modal.Volume.from_name("iris-run-e", create_if_missing=True)
VOLUME_PATH = "/output"

app = modal.App("iris-run-e", image=image)


# ── Main training function ─────────────────────────────────────────────────────
@app.function(
    gpu="A10G",                      # 24 GB VRAM; swap to "A100" if you want faster
    timeout=60 * 110,                # 110-minute ceiling (well above expected ~50 min)
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    cpu=4,
    memory=16384,
)
def run_e():
    import os
    import shutil
    import subprocess
    from pathlib import Path

    work = Path("/workspace")
    work.mkdir(exist_ok=True)

    # Make all scripts importable (needed by 3_evaluate.py → runs_utils)
    env = {**os.environ, "PYTHONPATH": "/scripts"}

    def sh(cmd: list[str]) -> None:
        """Run a command in /workspace, streaming output."""
        print(f"\n>>> {' '.join(str(c) for c in cmd)}\n{'─'*60}")
        subprocess.run(cmd, check=True, cwd=str(work), env=env)

    # ── 0. Download Kaggle dataset ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 0: Download Kaggle dataset")
    print("="*60)
    import kagglehub
    dataset_path = kagglehub.dataset_download("kellysanderson/myopia-image-dataset")
    images_dir = Path(dataset_path) / "IMAGES"
    print(f"\nDataset root:  {dataset_path}")
    print(f"IMAGES folder: {images_dir}")
    assert images_dir.exists(), f"Expected IMAGES/ folder at {images_dir}"

    # ── 1. CLAHE sampling (seed=42, 5k/class, 70/15/15) ──────────────────────
    print("\n" + "="*60)
    print("  STEP 1: CLAHE data split  (~10 min for 10k images)")
    print("="*60)
    sh([
        "python", "/scripts/1_sample_data.py",
        "--images-dir", str(images_dir),
        "--output-dir", "data_10k_clahe",
        "--n-per-class", "5000",
        "--clahe",
    ])

    # ── 2. Train YOLOv8n, 224px, 50 epochs ───────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2: Training run_E  (~30 min on A10G)")
    print("="*60)
    sh([
        "python", "/scripts/2_train.py",
        "--data",    "data_10k_clahe",
        "--model",   "yolov8n-cls.pt",
        "--imgsz",   "224",
        "--epochs",  "50",
        "--batch",   "64",            # A10G has 24 GB; 64 is safe for YOLOv8n
        "--patience", "15",
        "--name",    "run_E",
    ])

    # ── 3. Evaluate on held-out test set ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Evaluation on test set  (~10 min)")
    print("="*60)
    sh([
        "python", "/scripts/3_evaluate.py",
        "--run-dir",    "runs/classify/run_E",
        "--test-dir",   "data_10k_clahe/test",
        "--output-dir", ".",
    ])

    # ── 4. Persist outputs to Modal Volume ───────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4: Saving to volume 'iris-run-e'")
    print("="*60)

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

    # Ultralytics sometimes double-nests under runs/classify/classify/run_E
    # Search for the actual directory rather than hardcoding the path
    run_weight_hits = list((work / "runs").rglob("run_E/weights/best.pt"))
    if run_weight_hits:
        actual_run_dir = run_weight_hits[0].parent.parent
        print(f"  Found run dir at: {actual_run_dir}")
        _copy(actual_run_dir, out / "runs" / "classify" / "run_E")
    else:
        print(f"  WARNING: run_E/weights/best.pt not found under {work / 'runs'}")

    _copy(work / "figures" / "run_E",            out / "figures" / "run_E")
    _copy(work / "results" / "results_summary.csv", out / "results" / "results_summary.csv")

    # Flush volume writes so they're visible after the function exits
    volume.commit()

    print("\n" + "="*60)
    print("  run_E COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to Modal Volume 'iris-run-e' under {VOLUME_PATH}")
    print("\nDownload to Mac:")
    print("  modal volume get iris-run-e /output/runs/classify/run_E ./vinay-run/runs/classify/run_E")
    print("  modal volume get iris-run-e /output/figures/run_E       ./vinay-run/figures/run_E")
    print("  modal volume get iris-run-e /output/results/results_summary.csv ./vinay-run/results/results_summary_run_e.csv")


# ── Utility: list everything saved in the volume ──────────────────────────────
@app.function(volumes={VOLUME_PATH: volume})
def list_outputs():
    """List all files in the output volume with sizes."""
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


# ── Entry points ──────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    run_e.remote()
