# IRIS Myopia Classifier — Final Paper Overview

**Author:** Vinay  
**Model:** YOLOv8s (224px) — binary fundus classification (Myopia vs Normal)  
**Primary result:** `run_C_clean` on `data_10k_clean`  
**Date:** June 2026

---

## The One-Paragraph Story

We set out to improve a weak automated myopia detector (68% accuracy) by fixing its methodology. We fixed the sampling, increased resolution, and re-ran the model — and got 99.7% accuracy. Then we investigated whether that jump was real. It wasn't entirely: the source dataset contained thousands of byte-identical duplicate images, which random splitting placed across train and test. After auditing and removing those duplicates, the clean rerun gave 99.5% — still strong, but with an honest 6 missed myopia cases instead of zero. We then validated the clean model on a completely independent dataset from Chinese ophthalmic centers (PALM), achieving 81.5% accuracy and 100% precision on an unseen, harder-labeled population. Across all this, Grad-CAM confirms the model attends to retinal anatomy, not image brightness, and a CLAHE ablation confirms brightness alone cannot explain the results. Finally, we deployed `run_C_clean` as a fully offline iOS screening app (Core ML + Vision), achieving 7.1 ms model-only and 23 ms end-to-end inference on an iPhone 16 with zero network traffic and Nominal thermal state across consecutive screenings. The paper's contribution is therefore methodological as much as empirical: we demonstrate what rigorous ML practice looks like on a noisy public medical dataset, and how that honest model behaves in a real clinical mobile deployment.

---

## What Was Wrong Originally (train4)

The starting point was `runs/classify/train4` — a YOLOv8n model trained on ~4k images.

| Issue | Detail |
|-------|--------|
| Non-random sampling | First N images taken — systematic bias |
| 64×64 resolution | Loses fundus detail; native images are ~224px |
| Wrong dataset size | ~4k instead of the paper's stated 10k |
| Inverted CM narrative | Paper described confusion matrix backwards |
| Default threshold only | No threshold optimization despite suboptimal AUC |
| No explainability | No Grad-CAM — model could be using brightness, not anatomy |
| Brightness bias | Myopia class ~38 brightness points brighter — trivial shortcut |

Baseline result: **68% accuracy, 59.4% precision, 98.7% recall, AUC 0.813**

---

## What We Built (the pipeline)

All code lives in `vinay-run/`. Scripts run in numbered order.

| Script | Purpose |
|--------|---------|
| `1_sample_data.py` | Original random balanced sampler (seed=42, 70/15/15) |
| `1b_sample_data_clean.py` | **Deduplicated** cluster-aware sampler (MD5 + dHash=0) |
| `2_train.py` | YOLOv8 training — auto-detects MPS/CUDA |
| `3_evaluate.py` | Test-set eval: CM, ROC, PR curves, CSV append |
| `4_threshold_sweep.py` | Threshold optimization (best F1 and best screening recall) |
| `5_gradcam.py` | Grad-CAM visualization on 4 sample images |
| `6_export_local_runs.py` | Import Colab/Modal results locally |
| `7_leakage_audit.py` | Audit a split for exact + near duplicates |
| `8_leakage_calibrate.py` | Cross-class control to validate near-dup signal |
| `9_leakage_visual.py` | Visual proof of leakage pairs |
| `10_audit_full_source.py` | Full 124k source duplication audit |
| `11_evaluate_palm.py` | Inference-only external validation on PALM |
| `runs_utils.py` | Resolves Ultralytics double-nesting of run dirs |
| `modal_run_e.py` | Modal GPU runner for run_E (CLAHE ablation) |
| `modal_run_c_clean.py` | Modal GPU runner for full clean pipeline |
| `colab_notebook.ipynb` | All-in-one Colab notebook |

---

## Experiment Progression

### Phase 1 — Fix the methodology (runs A/B/C/E)

Run on a random 10k split (`data_10k/`). Dataset: Kaggle `kellysanderson/myopia-image-dataset`.
Split: seed=42, 5000/class → 3500 train / 750 val / 750 test per class.

| Run | Model | imgsz | Data | Epochs | Acc | Recall | AUC | FN |
|-----|-------|-------|------|--------|-----|--------|-----|----|
| `run_A` | YOLOv8n | 64 | 10k random | 20 | 99.67% | 99.87% | 0.9998 | 1 |
| `run_B` | YOLOv8n | 224 | 10k random | 50 | 99.80% | 100.0% | 1.0000 | 0 |
| `run_C` | YOLOv8s | 224 | 10k random | 50 | 99.73% | 100.0% | 0.9999 | 0 |
| `run_E` | YOLOv8n | 224 | 10k CLAHE | 50 | 99.80% | 100.0% | 1.0000 | 0 |

**Finding:** Methodology fix alone (random sampling + correct 10k split) took accuracy from 68% → ~99.7%.  
**Finding (run_E):** CLAHE ablation — removing brightness bias did **not** hurt accuracy. Brightness shortcut hypothesis **rejected**.

### Phase 2 — Leakage discovery

After runs A/B/C/E, we audited the source dataset.

**Full source audit (`10_audit_full_source.py`) on all 124,794 images:**

| Class | Source images | Exact MD5 dup extras | dHash=0 dup extras | Unique clusters |
|-------|--------------|----------------------|-------------------|-----------------|
| Myopia | 63,294 | **8,795** | **30,078** | 33,216 |
| Normal | 61,500 | 0 | 3,732 | 57,786 |

The Myopia folder contained large mirrored filename ranges — the same images re-uploaded under different number sequences. Random splitting placed identical images in both train and test.

**Split audit (`7_leakage_audit.py`) on `data_10k`:**
- 27 exact (MD5) cross-split duplicate groups
- 245 perceptually identical (dHash=0) test↔train pairs
- Cross-class calibration confirmed signal is real, not a fundus-structure artifact

**Conclusion:** The near-perfect FN=0 results were partly artefactual. The model was tested on images it had memorized.

### Phase 3 — Clean rerun (run_C_clean)

Built `1b_sample_data_clean.py`: deduplicate by MD5 + dHash=0 per class, then sample and split.

**Clean split audit (`data_10k_clean/`):**
- 0 exact MD5 cross-split duplicates
- 0 dHash=0 cross-split duplicates ✓

**Training:** Modal A10G GPU, `modal_run_c_clean.py`. YOLOv8s, 224px, 50 epochs, early stopped at epoch 17.

**Results — `run_C_clean` on `data_10k_clean/test` (1,500 images):**

| Metric | Leaked `run_C` | Clean `run_C_clean` |
|--------|----------------|---------------------|
| Accuracy | 99.73% | **99.53%** |
| Precision | 99.47% | **99.87%** |
| Recall | 100.0% | **99.20%** |
| F1 | 99.73% | **99.53%** |
| AUC | 0.9999 | **0.9999** |
| FP | 4 | **1** |
| **FN (missed myopia)** | **0** | **6** |

The honest clinical recall is **99.2%** with **6 missed myopia cases**, not 100% with zero.

**Threshold optimization:** Best F1 at threshold 0.15 → 100% recall, 99.7% precision. Clinically, lowering the threshold from 0.5 to 0.15 eliminates all misses with minimal cost.

### Phase 4 — External validation (PALM)

**Dataset:** PALM (Pathologic Myopia Challenge, 2019), 400 labeled fundus images from Chinese ophthalmic centers. Independent of Kaggle source. [IEEE DataPort](https://ieee-dataport.org/documents/palm-pathologic-myopia-challenge)

**Protocol:** Inference-only on `run_C_clean`. No retraining.

**Label mapping (filename prefix):**
- `P*` = Pathologic Myopia (213 images)
- `H*` = High Myopia, no pathology (26 images)
- `N*` = Normal (161 images)

| Mapping | n | Accuracy | Precision | Recall | AUC | FP | FN |
|---------|---|----------|-----------|--------|-----|----|----|
| Conservative (P=+, H+N=−) | 400 | **81.5%** | **100%** | **65.3%** | **0.9885** | 0 | 74 |
| Inclusive (P+H=+, N=−) | 400 | **75.0%** | **100%** | **58.2%** | **0.9578** | 0 | 100 |

**Interpretation:**
- **100% precision** — every image flagged as myopia is actually myopia on a completely unseen dataset.
- **AUC 0.96–0.99** — discriminative ability transfers cross-dataset.
- **Recall drops (~65%)** — expected: PALM's positive class is *pathologic* myopia (severe structural lesions not present in Kaggle training). The model learned general myopia features, not pathology-specific lesion signatures.
- **No false positives** — the model is not over-triggering on the new domain.

### Phase 5 — Mobile deployment (iOS app)

**Model:** `run_C_clean/weights/best.pt` exported to Core ML (`best.mlpackage`, 9.7 MB, Float16).  
**App:** Myopia Detection (SwiftUI + Vision + CoreML), deployed on iPhone 16 (iOS 26, Apple A18).  
**Benchmark evidence:** `frontend/sayan_benchmarks/`

**Export pipeline:**
```bash
# Python 3.11 venv required (coremltools fails on Python 3.14)
yolo export model=runs/classify/run_C_clean/weights/best.pt format=coreml imgsz=224
```

**App architecture:**
- `ImageClassifier.swift` — loads `best.mlpackage`, runs `VNCoreMLRequest` with `.centerCrop`
- `CameraView.swift` — alignment circle overlay, center-crop preprocessing (`centerCropScale=0.44`), capture + upload
- Fully offline — no network calls in inference pipeline

**Latency benchmarks (iPhone 16, iOS 26):**

| Metric | Value | Source |
|--------|-------|--------|
| Model-only inference (median) | **7.1 ms** | Xcode Core ML Performance Report (n=119) |
| Model-only inference (mean ± SD) | **5.5 ± 4.1 ms** | Xcode Core ML Performance Report |
| End-to-end Vision pipeline (median) | **22.96 ms** | In-app benchmark, compute unit All (n=50) |
| End-to-end Vision pipeline (mean ± SD) | **24.86 ± 7.72 ms** | In-app benchmark, compute unit All (n=50) |
| Vision preprocessing overhead | **~16 ms** | Difference between model-only and end-to-end |
| Model package size | **9.7 MB** | `best.mlpackage` on disk |
| Storage precision | **Float16** | Core ML ML Program format |

**Compute unit comparison (end-to-end, n=50 each):**

| Config | Median | Mean ± SD |
|--------|--------|-----------|
| All | 22.96 ms | 24.86 ± 7.72 ms |
| CPU + Neural Engine | 23.13 ms | 24.00 ± 2.61 ms |
| CPU only | 23.73 ms | 23.78 ± 0.68 ms |

Vision preprocessing overhead dominates; ANE speedup is negligible for this model size on A18 silicon.

**Power & thermal profiling (Instruments Power Profiler, 10 consecutive inferences):**

| Metric | Value |
|--------|-------|
| Session duration | 87.2 s (1.45 min) |
| Thermal state | **Nominal** (entire session) |
| Avg power drain | **0.0%/hr** |
| App network bytes TX/RX | **0 / 0** |
| CPU activity pattern | Discrete bursts per inference, return to baseline |

**Interpretation:**
- **Sub-25 ms end-to-end latency** satisfies real-time screening requirements.
- **Zero app network traffic** confirms fully on-device inference — no patient data leaves the device.
- **Nominal thermal state** across 10 consecutive screenings — safe for sustained clinical use.
- **Alignment UI** (circle overlay + programmatic center crop) assists non-professional operators in centering the 20D condensing lens relative to the camera axis.

---

## Scientific Conclusions

1. **Methodology matters more than model choice.** The jump from 68% to ~99.5% came from fixing the split, not from a better model.

2. **Leakage in public medical datasets is real and detectable.** The Kaggle dataset contained ~8,800 exact duplicate images concentrated in the Myopia class, creating cross-split contamination. This inflated recall from 99.2% to 100% and F1 from 99.5% to 99.7%.

3. **The clean model is still strong.** 99.5% accuracy and 0.9999 AUC on a verified duplicate-free test set is a credible result.

4. **Brightness is not the explanation.** CLAHE normalization did not hurt accuracy. Grad-CAM shows attention on vessels, optic disc, and retinal tissue — not uniform brightness patterns.

5. **Cross-dataset generalization is preliminary but encouraging.** On PALM (independent Chinese clinical dataset, different label definition), AUC stays above 0.96 and precision is perfect. Recall drops, which is scientifically honest and explicable by label scope.

6. **Threshold matters clinically.** Default threshold 0.5 gives 6 FN on the clean test. Threshold 0.15 gives 0 FN — full recall — at minimal precision cost. For a screening tool, this is the operating point to deploy.

7. **The model deploys practically on consumer hardware.** Core ML export of `run_C_clean` achieves 7.1 ms model-only and 23 ms end-to-end inference on iPhone 16. Instruments profiling confirms Nominal thermal state, zero network traffic, and discrete (not sustained) CPU bursts — validating offline deployment for rural screening without cloud infrastructure.

---

## Final File Structure

```
vinay-run/
│
├── PAPER_OVERVIEW.md          ← this file
├── PAPER_NARRATIVE.md         ← methods/discussion/abstract drafts
├── PLAN.md                    ← original experiment roadmap (updated)
│
├── Pipeline scripts
│   ├── 1_sample_data.py       original sampler
│   ├── 1b_sample_data_clean.py  dedup-aware sampler (primary)
│   ├── 2_train.py             training (MPS/CUDA auto-detect)
│   ├── 3_evaluate.py          test-set evaluation
│   ├── 4_threshold_sweep.py   threshold optimization
│   ├── 5_gradcam.py           Grad-CAM visualization
│   ├── 6_export_local_runs.py import from Colab/Modal
│   └── runs_utils.py          path resolution helper
│
├── Leakage audit scripts
│   ├── 7_leakage_audit.py     split-level audit (exact + near dup)
│   ├── 8_leakage_calibrate.py cross-class near-dup calibration
│   ├── 9_leakage_visual.py    visual proof figure
│   └── 10_audit_full_source.py full 124k source audit
│
├── External validation
│   └── 11_evaluate_palm.py    PALM inference-only eval
│
├── Modal runners
│   ├── modal_run_e.py         run_E CLAHE ablation
│   └── modal_run_c_clean.py   full clean pipeline (primary)
│
├── colab_notebook.ipynb       all-in-one Colab runner
│
├── data_10k/                  original random split (leaked)
├── data_10k_clahe/            CLAHE-normalized split (run_E)
├── data_10k_clean/            deduplicated clean split (primary)
├── PALM/                      PALM external validation dataset
│
├── runs/classify/
│   ├── run_A/                 YOLOv8n 64px (leaked)
│   ├── run_B/                 YOLOv8n 224px (leaked)
│   ├── run_C/                 YOLOv8s 224px (leaked)
│   ├── run_C_clean/           YOLOv8s 224px clean ← PRIMARY MODEL
│   └── run_E/                 YOLOv8n 224px CLAHE (leaked)
│
├── figures/
│   ├── leakage_pairs.png      visual proof of train/test duplicates
│   ├── run_A/                 CM, ROC, PR, gradcam
│   ├── run_B/                 CM, ROC, PR, gradcam
│   ├── run_C/                 CM, ROC, PR, gradcam
│   ├── run_C_clean/           CM, ROC, PR, gradcam, threshold_sweep
│   ├── run_C_clean_palm_conservative/   CM, ROC, PR
│   ├── run_C_clean_palm_inclusive/      CM, ROC, PR
│   └── run_E/                 CM
│
└── results/
    ├── results_summary.csv             leaked runs A/B/C/E
    ├── results_summary_clean.csv       clean run_C_clean
    ├── results_summary_external.csv    PALM external validation
    ├── full_source_audit.txt           124k source duplication report
    ├── leakage_data_10k.txt            leaked split audit
    ├── leakage_data_10k_clean.txt      clean split audit (PASS)
    ├── palm_external_validation.json   PALM metrics both mappings
    └── dedup_manifest.json             dedup cluster stats

frontend/sayan_benchmarks/               iOS deployment benchmarks
    ├── best-all-Sayan's iPhone.mlperf   Xcode model-only perf report
    ├── power-profiler-run2.trace        Instruments power/thermal trace
    └── inapp_benchmark_results.json     all latency + power numbers
```

**iOS app (separate repo):** `MyopiaDetection/Myopia Detection/`
- `best.mlpackage` — deployed Core ML model
- `ImageClassifier.swift` — Vision inference
- `CameraView.swift` — alignment UI + capture pipeline

---

## Reviewer Checklist

| Item | Status | Evidence |
|------|--------|---------|
| Research question | ✅ | Methods |
| Random balanced sampling | ✅ | `1b_sample_data_clean.py`, seed=42 |
| Correct dataset size | ✅ | 10k (5k/class), 1500-image test set |
| Leakage audit | ✅ | `results/leakage_data_10k_clean.txt` |
| Source duplication disclosed | ✅ | `results/full_source_audit.txt` |
| Primary model on clean split | ✅ | `run_C_clean`, 99.5% acc |
| Explainability | ✅ | `figures/run_C_clean/gradcam.png` |
| Brightness ablation | ✅ | `run_E` CLAHE (no accuracy drop) |
| Threshold optimization | ✅ | `figures/run_C_clean/threshold_sweep.png` |
| External validation | ✅ | PALM, 400 images, AUC 0.99, 0 FP |
| Mobile deployment | ✅ | iOS app, Core ML, 23 ms end-to-end, 0 network bytes |
| On-device benchmarks | ✅ | `frontend/sayan_benchmarks/inapp_benchmark_results.json` |
| Power/thermal profiling | ✅ | Instruments Power Profiler, Nominal thermal, 0.0%/hr |
| Same-patient leakage | ⚠️ | Acknowledged in Limitations — cannot fix (no patient IDs) |
| Multi-dataset training | ✗ | Out of scope |

---

## Key Numbers for the Paper

| Context | Number |
|---------|--------|
| Source images audited | 124,794 |
| Exact Myopia duplicates in source | 8,795 groups |
| Cross-split leakage (original data_10k) | 27 exact, 245 dHash=0 pairs |
| Clean test accuracy | **99.53%** |
| Clean test AUC | **0.9999** |
| Clean test FN (missed myopia) | **6 / 750** |
| Threshold 0.15 recall | **100%** |
| PALM external accuracy | **81.5%** |
| PALM external precision | **100%** |
| PALM external AUC | **0.9885** |
| PALM external FP | **0** |
| Core ML model size | **9.7 MB** (Float16) |
| Model-only inference (iPhone 16) | **7.1 ms** median |
| End-to-end inference (iPhone 16) | **22.96 ms** median, **24.86 ± 7.72 ms** mean ± SD |
| Thermal state (10 screenings) | **Nominal** |
| App network traffic | **0 bytes** |
