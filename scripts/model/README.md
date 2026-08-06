# `model/` — Phase 4: training

Fine-tune **RF-DETR** on the unified COCO dataset, plus a **YOLO** baseline on the identical data.

- `train_rfdetr.py` (planned): `pip install rfdetr` (Python ≥3.10, CUDA). Start `RFDETRMedium` (576),
  then `RFDETRLarge` (704). Resolution divisible by 56; effective batch 16 (`batch_size=4,
  grad_accum_steps=4` on 16 GB). Report from `checkpoint_best_ema.pth`.
  ⚠️ Logging extra is `rfdetr[loggers]` (there is no `rfdetr[metrics]`). Stay within Nano–Large (Apache-2.0).
- `train_yolo.py` (planned): `ultralytics` YOLO v11/v12 on the same COCO (converted to YOLO) — the
  credible same-data comparison, since YOLO can rival RF-DETR on the smallest objects.

Dataset must be COCO with `train/ valid/ test/` folders (Phase 3 produces these). Log every run in
`../EXPERIMENTS.md`.
