# Vinay Run — Experiment Plan

## Goal
Improve on the original `train4` results (68% acc, 59.4% precision, 98.7% recall, 77.5% F1)
by fixing methodological issues and running a structured ablation in Google Colab.

**Update (leakage recovery):** Initial A/B/C/E runs on `data_10k` are superseded by clean-split
experiments on `data_10k_clean`. See `PAPER_NARRATIVE.md` for the revised paper story.

---

## Leakage Recovery (Current Priority)

| Step | Script | Output |
|------|--------|--------|
| Deduplicated sampling | `1b_sample_data_clean.py` | `data_10k_clean/` |
| Split leakage audit | `7_leakage_audit.py` | `results/leakage_data_10k_clean.txt` |
| Full source audit | `10_audit_full_source.py` | `results/full_source_audit.txt` |
| Primary clean model | `modal run modal_run_c_clean.py` or `2_train.py --name run_C_clean` | `runs/classify/run_C_clean/` |
| Clean evaluation | `3_evaluate.py --results-csv results_summary_clean.csv` | `results/results_summary_clean.csv` |

Acceptance criteria for clean split: **0** exact MD5 cross-split dupes, **0** dHash=0 cross-split dupes.

---

## What Was Wrong With the Original

| Issue | Detail |
|-------|--------|
| Non-random sampling | First 5k images taken per class — not representative |
| Low resolution | 64×64 destroys fundus detail (native images are 224×224) |
| Wrong dataset | `train4` used a different ~4k dataset, not the paper's 10k split |
| Inverted CM narrative | Paper text describes confusion matrix results backwards |
| Single threshold | Default 0.5 — ROC AUC 0.813 suggests a better threshold exists |
| No explainability | No evidence model looks at retinal anatomy, not just brightness |

---

## Experiment Matrix

Run these in order. Each builds on the previous.

| Run ID | Model | imgsz | Data | Epochs | Purpose |
|--------|-------|--------|------|--------|---------|
| `run_A` | YOLOv8n | 64 | 10k random | 20 | Reproduce paper baseline |
| `run_B` | YOLOv8n | 224 | 10k random | 50 | Resolution upgrade |
| `run_C` | YOLOv8s | 224 | 10k random | 50 | Model upgrade |
| `run_D` | YOLOv8s | 224 | 20k random | 50 | More data |

Compare all runs in final results table. Report on the best model.

---

## Scripts (run in order)

### Step 1 — `1_sample_data.py`
- Randomly samples N images per class from `IMAGES/` (seed=42)
- Creates a YOLO-compatible folder split:
  ```
  data/
    train/Myopia/  train/Normal/
    validation/Myopia/  validation/Normal/
    test/Myopia/  test/Normal/
  ```
- Run once for 10k (5k/class), once for 20k (10k/class)
- Outputs: `data_10k/` and `data_20k/`

### Step 2 — `2_train.py`
- Accepts `--model`, `--imgsz`, `--data`, `--epochs`, `--run-name`
- Runs one training experiment, saves to `runs/classify/<run-name>/`
- Use this once per experiment in the matrix above

### Step 3 — `3_evaluate.py`
- Loads a trained model from a run directory
- Evaluates on the held-out **test set** (not validation)
- Outputs:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix (correct orientation)
  - ROC curve + AUC
  - Precision-Recall curve + AP

### Step 4 — `4_threshold_sweep.py`
- Loads softmax probabilities from the best model on test set
- Sweeps thresholds 0.05 → 0.95
- Plots F1 vs threshold, finds optimal
- Reports optimized-threshold metrics alongside default (0.5)

### Step 5 — `5_gradcam.py`
- Runs Grad-CAM on 4 sample images (2 Myopia, 2 Normal)
- Overlays heatmap on original fundus image
- Saves to `vinay-run/figures/gradcam_examples.png`
- Used as Figure in paper to show model attends to retinal anatomy

---

## Results to Report in Paper

### Main Table (from Step 3)

| Run | Model | imgsz | Data | Acc | Prec | Recall | F1 | AUC |
|-----|-------|--------|------|-----|------|--------|-----|-----|
| run_A | YOLOv8n | 64 | 10k | ? | ? | ? | ? | ? |
| run_B | YOLOv8n | 224 | 10k | ? | ? | ? | ? | ? |
| run_C | YOLOv8s | 224 | 10k | ? | ? | ? | ? | ? |
| run_D | YOLOv8s | 224 | 20k | ? | ? | ? | ? | ? |

### Threshold Table (from Step 4, best model only)

| Threshold | Acc | Prec | Recall | F1 |
|-----------|-----|------|--------|-----|
| 0.50 (default) | ? | ? | ? | ? |
| Best F1 threshold | ? | ? | ? | ? |
| Best Recall threshold | ? | ? | ? | ? |

---

## Paper Sections to Update After Reruns

| Section | What to fix |
|---------|-------------|
| **Abstract** | Update all 4 metrics to best run results |
| **Methods → Data** | Note random seed=42, document split procedure |
| **Methods → Training** | Add resolution comparison rationale |
| **Results → Confusion Matrix** | Fix inverted narrative (987 Myopia correctly classified, 373 Normal correctly classified — not the other way round) |
| **Results → Add new table** | Model + resolution comparison table |
| **Discussion** | Fix "high false negatives for myopia" → should be "high false positives (627 normals flagged as myopic)" |
| **Discussion** | Add threshold optimization finding |
| **Limitations** | Add brightness artifact discussion (systematic brightness gap between classes) |
| **Future Work** | Update to reflect what was actually done |

---

## Confusion Matrix — Correct Interpretation

The original `matrix.py` numbers (from train4's holdout evaluation):
```
              Pred Normal    Pred Myopia
True Normal       373            627     ← 627 normals wrongly flagged
True Myopia        13            987     ← only 13 myopia cases missed
```

- **False negatives (missed myopia): 13** — very low, good for screening
- **False positives (normals flagged as myopic): 627** — high, needs discussion
- The paper text says the opposite of this — must be corrected

---

## Colab Setup

Upload to Google Drive:
```
MyDrive/
  IRIS-data/
    IMAGES/
      Myopia_images/   (63,294 images)
      Normal_images/   (61,500 images)
  vinay-run/           (copy this folder)
```

Then open `colab_notebook.ipynb` and run top to bottom.
GPU: T4 (free tier) is sufficient for all experiments.
Estimated total GPU time: ~2 hours for all 4 runs.

---

## File Structure

```
vinay-run/
  PLAN.md                  ← this file
  1_sample_data.py         ← build train/val/test splits
  2_train.py               ← training script
  3_evaluate.py            ← full evaluation on test set
  4_threshold_sweep.py     ← threshold optimization
  5_gradcam.py             ← Grad-CAM visualization
  colab_notebook.ipynb     ← all-in-one Colab notebook
  figures/                 ← output figures (created at runtime)
  results/                 ← CSV outputs (created at runtime)
```
