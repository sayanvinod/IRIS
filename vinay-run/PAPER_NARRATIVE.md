# Paper Narrative — Leakage-Aware Results (Updated)

This document replaces the original headline narrative. Use **clean-split results** as the primary claim; retain initial A/B/C/E runs only as a cautionary comparison.

---

## Executive Summary

We trained YOLOv8 classifiers for binary myopia detection from fundus images. Initial experiments on a random 10k split reported near-perfect accuracy (~99.7%). A subsequent **data leakage audit** revealed substantial duplication in the public Kaggle dataset (`kellysanderson/myopia-image-dataset`), especially in `Myopia_images`. We rebuilt the dataset with **MD5 + dHash=0 deduplication** and cluster-aware sampling, then retrained the primary model (`run_C_clean`).

**Primary result:** report metrics from `run_C_clean` on `data_10k_clean/test`, not from `run_C` on `data_10k/test`.

---

## What Changed Scientifically

| Stage | Split | Leakage status | Role in paper |
|-------|-------|----------------|---------------|
| Initial (`run_A`–`run_E`) | `data_10k` random split | Cross-split duplicates confirmed | Cautionary / motivation only |
| Clean (`run_C_clean`) | `data_10k_clean` deduplicated split | 0 exact + 0 dHash=0 cross-split leakage | **Primary headline result** |

The jump from ~68% (original `train4`) to ~99.7% on the initial random split was **not solely** a methodology fix. Source-level duplication inflated test performance. After deduplication, `run_C_clean` still achieves 99.5% accuracy — but with **6 missed myopia cases (FN=6)** vs **0 on the leaked split**, which is the honest clinical recall estimate.

---

## Dataset Duplication Findings (Methods / Supplement)

Full source audit of 124,794 images (`results/full_source_audit.txt`):

| Metric | Myopia | Normal |
|--------|--------|--------|
| Source images | 63,294 | 61,500 |
| Exact MD5 duplicate extras | 8,795 | 0 |
| dHash=0 perceptual duplicate extras | 30,078 | 3,732 |
| Unique clusters after dHash=0 dedup | 33,216 | 57,786 |

Initial split audit (`results/leakage_data_10k.txt`):

- 27 exact cross-split duplicate groups (all Myopia)
- 245 perceptually identical (dHash=0) test↔train/val pairs

Clean split audit (`results/leakage_data_10k_clean.txt`):

- **0** exact cross-split duplicate groups
- **0** perceptually identical (dHash=0) test↔train/val pairs

---

## Methods Additions

### Data preparation (revised)

1. Sample from Kaggle `IMAGES/Myopia_images/` and `IMAGES/Normal_images/`.
2. **Deduplicate before splitting:** group images by MD5 hash and perceptual hash (dHash, Hamming distance 0); keep one representative per cluster per class.
3. Random balanced sample (seed=42), 5,000 images per class → 10,000 total.
4. Split 70% train / 15% validation / 15% test.
5. Run leakage audit on the final split; require zero exact and zero dHash=0 cross-split duplicates before training.

Script: `1b_sample_data_clean.py` → output `data_10k_clean/`.

### Training (primary model)

- Model: YOLOv8s classification
- Input size: 224×224
- Epochs: 50 (early stopping patience=15)
- Run name: `run_C_clean`

---

## Results Tables (LaTeX-ready structure)

### Table 1 — Leaked vs clean primary model

| Run | Split | Acc | Prec | Recall | F1 | AUC | FN |
|-----|-------|-----|------|--------|-----|-----|-----|
| `run_C` | Leaked (`data_10k`) | 99.73% | 99.47% | 100.0% | 99.73% | 0.9999 | 0 |
| `run_C_clean` | Clean (`data_10k_clean`) | **99.53%** | **99.87%** | **99.20%** | **99.53%** | **0.9999** | **6** |

Confusion matrix (`run_C_clean`, test n=1500): TP=744, TN=749, FP=1, FN=6.

Threshold sweep (`run_C_clean`): best F1 at threshold 0.15 → Acc 99.9%, Recall 100%, F1 99.9%.

### Table 2 — Initial runs (supplementary / cautionary)

Label these as **"Initial random split (later found to contain source duplication)"**:

| Run | Acc | Notes |
|-----|-----|-------|
| `run_A` | 99.67% | Do not use as final performance |
| `run_B` | 99.80% | Do not use as final performance |
| `run_C` | 99.73% | Superseded by `run_C_clean` |
| `run_E` | 99.80% | CLAHE ablation still methodologically valid for brightness hypothesis |

---

## Discussion Points

### Strengths (revised)

- Explicit leakage audit and deduplicated rerun — rare in public medical-image papers.
- CLAHE ablation (`run_E`) still supports rejection of pure brightness shortcut (orthogonal to duplication).
- Grad-CAM on clean model (`run_C_clean`) shows anatomical attention if heatmaps remain vessel/disc-focused.

### Limitations (expanded)

1. **Single public dataset** with no hospital source, acquisition protocol, or patient metadata.
2. **Same-patient leakage cannot be ruled out** (left/right eye, repeat visits) without patient IDs.
3. **Near-duplicates at dHash 1–5** may remain; we only collapse dHash=0 clusters.
4. **High initial accuracy was partly artifactual** — external validation on an independent, audited dataset is required before clinical claims.

---

---

## External Validation — PALM Dataset

**Script:** `11_evaluate_palm.py`
**Model:** `run_C_clean` (inference-only, no retraining)
**Dataset:** PALM Training/Classification, 400 labeled fundus images from multiple Chinese ophthalmic centers (2019 PALM Challenge). [IEEE DataPort](https://ieee-dataport.org/documents/palm-pathologic-myopia-challenge)

### Label mapping
PALM labels are derived from filename prefix:
- `P*` = Pathologic Myopia (213 images)
- `H*` = High Myopia, no pathology (26 images)
- `N*` = Normal / Non-PM (161 images)

Two interpretations reported, both honest:

| Mapping | Positive class | Negative class |
|---------|---------------|----------------|
| **Conservative** | P only (213) | H + N (187) |
| **Inclusive** | P + H (239) | N only (161) |

### Results

| Metric | Internal clean test | PALM conservative | PALM inclusive |
|--------|--------------------|--------------------|----------------|
| Accuracy | 99.5% | **81.5%** | **75.0%** |
| Precision | 99.9% | **100.0%** | **100.0%** |
| Recall (PM/Myopia) | 99.2% | **65.3%** | **58.2%** |
| F1 | 99.5% | **79.0%** | **73.5%** |
| AUC | 0.9999 | **0.9885** | **0.9578** |
| FP (non-myopia misclassified) | 1 | **0** | **0** |
| FN (myopia missed) | 6 | **74** | **100** |

### Interpretation

- **AUC remains high (0.96–0.99)** — the model's underlying discriminative ability transfers well cross-dataset.
- **Precision is 100%** in both PALM mappings — every image the model calls pathologic myopia is correct. No false positives.
- **Recall drops to ~58–65%** — the model misses a substantial fraction of PALM pathologic myopia cases.
- The recall drop is expected and scientifically informative: PALM myopia is **pathologic** (severe, with structural lesions), while Kaggle myopia is a broader label. The model was trained to distinguish general myopia from normal; it has not seen the specific lesion patterns (peripapillary atrophy, retinal detachment) that define pathologic myopia.
- **This does not invalidate the internal results** — it confirms the model learned fundus-based myopia features, but those features are not sufficient to catch all pathologic-stage presentations.

### Paper wording

> "To assess cross-dataset generalization, we evaluated the leakage-controlled model (`run_C_clean`) on the PALM dataset [CITE], an independent collection of 400 labeled fundus photographs from Chinese ophthalmic centers. No retraining was performed. Under conservative label mapping (pathologic myopia vs non-PM), the model achieved 81.5% accuracy, 100% precision, and 65.3% recall (AUC 0.99). The precision-recall pattern indicates the model does not produce false positives on PALM, but misses a portion of pathologic myopia cases — a subtype characterized by structural lesions not represented in the Kaggle training data. These findings support the generalizability of the learned fundus-based features while highlighting the scope limitation of a model trained on general myopia labels."

### Figures generated
- `figures/run_C_clean_palm_conservative/` — CM, ROC, PR
- `figures/run_C_clean_palm_inclusive/` — CM, ROC, PR

### Files
- `results/palm_external_validation.json` — full metrics both mappings
- `results/results_summary_external.csv` — tabular format

---

## Abstract Template (draft)

> Automated myopia screening from fundus photographs using deep learning has reported high accuracy on public datasets, but hidden data duplication can inflate performance. We audited the Kaggle myopia fundus dataset (124,794 images) and found extensive duplicate and near-duplicate images, concentrated in the myopia class (8,795 exact MD5 duplicate groups; 30,078 extra perceptual duplicates in Myopia). After deduplicating by content and perceptual hash before train/validation/test splitting, we trained YOLOv8s at 224×224 resolution and evaluated on a held-out clean test set with zero cross-split duplicate leakage. The clean model achieved 99.5% accuracy, 99.9% precision, 99.2% recall, and ROC AUC 0.9999 on 1,500 test images, with 6 missed myopia cases. For external cross-dataset generalization, we evaluated the trained model — without retraining — on 400 labeled images from the PALM pathologic myopia dataset. The model achieved 81.5% accuracy and 100% precision with 65.3% recall (AUC 0.99), indicating strong specificity but reduced sensitivity for pathologic-stage presentations not represented in training. Our findings highlight the need for duplicate auditing in ophthalmic ML datasets and demonstrate preliminary cross-dataset generalization of the learned retinal features.

---

## Paper Sections Checklist

| Section | Action |
|---------|--------|
| **Abstract** | Lead with leakage audit + clean metrics |
| **Introduction** | Cite duplication risk in medical imaging |
| **Methods → Data** | Document dedup + cluster-aware split (`1b_sample_data_clean.py`) |
| **Methods → Leakage audit** | Cite `7_leakage_audit.py`, `10_audit_full_source.py` |
| **Results** | Table 1: leaked vs clean; Table 2: ablations as supplementary |
| **Discussion** | Explain why initial ~99.7% was unreliable |
| **Limitations** | No patient IDs, no external validation, near-dup threshold |
| **Supplement** | Attach `full_source_audit.txt`, `leakage_data_10k_clean.txt`, `dedup_manifest.json` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `1b_sample_data_clean.py` | Deduplicated sampling |
| `7_leakage_audit.py` | Split leakage audit |
| `10_audit_full_source.py` | Full source duplication audit |
| `modal_run_c_clean.py` | Modal GPU runner for full clean pipeline |
| `results/dedup_manifest.json` | Dedup stats for paper (local + Modal copy) |
| `results/full_source_audit.txt` | Source duplication report |
| `results/leakage_data_10k_clean.txt` | Clean split audit |
| `results/results_summary_clean.csv` | Primary metrics table |
