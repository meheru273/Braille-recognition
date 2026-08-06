# `eval/` — Phase 5: evaluation

Two layers, on a **page-level** test split (never split cells from the same page — leakage).

1. **Detection metrics:** headline mAP@0.5:0.95 + mAP@0.5, AP-small, precision/recall/F1, per-class AP,
   PR curves, confusion matrix.
   - Baseline (hidden Roboflow model): `supervision.metrics` (minimal glue).
   - Per-epoch RF-DETR val: `torchmetrics MeanAveragePrecision(class_metrics=True)`.
   - Final reported number: `pycocotools` — ⚠️ raise `params.maxDets` (e.g. `[100,300,1000]`) for dense pages.
2. **End-to-end reading:** assemble cells → reading order → back-translate with **liblouis** →
   **CER/WER** via **jiwer**, reported Grade-1 vs Grade-2 separately; ablate reading-order with oracle ordering.

Comparisons: Roboflow baseline vs RF-DETR (Medium/Large) vs YOLO. Ablations: crop on/off, resolution,
pretrained vs scratch, augmentation, train-set-size learning curve. Record everything in `../EXPERIMENTS.md`.
