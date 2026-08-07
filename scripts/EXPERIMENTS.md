# Experiment Log

Append one row per training/eval run so the paper's numbers stay reproducible.
All metrics on the frozen **page-level** test split.

## Detection

| Date | Run ID | Model | Res | Data (imgs, split) | Epochs | mAP@.5 | mAP@.5:.95 | AP-small | P / R / F1 | Notes |
|------|--------|-------|-----|--------------------|--------|--------|-----------|----------|-----------|-------|
| _tbd_ | baseline-roboflow | Roboflow hosted | n/a | test | n/a | | | | | scored as-is via supervision.metrics |
| 2026-08-07 | rfdetr-medium-v1 | RFDETRMedium | 576 | base 444 (seed-8 split) | 50 | 0.456* | 0.289* | | F1 0.548 / P 0.653 / R 0.501* | *VAL epoch 48, still improving; regular ckpt (EMA lagged ~0.02); console tables not persisted this run — tee added after. Report TEST from checkpoint_best_ema.pth when run finishes. |
| _tbd_ | yolo11-v1 | YOLO11 | | | | | | | | same-data baseline |

## End-to-end reading (CER/WER)

| Run ID | Grade | cell-level CER | text-level CER | WER | reading-order | Notes |
|--------|-------|----------------|----------------|-----|---------------|-------|
| _tbd_ | G1 | | | | predicted / oracle | liblouis back-translation |
