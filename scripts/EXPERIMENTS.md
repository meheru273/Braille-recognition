# Experiment Log

Append one row per training/eval run so the paper's numbers stay reproducible.
All metrics on the frozen **page-level** test split.

## Detection

| Date | Run ID | Model | Res | Data (imgs, split) | Epochs | mAP@.5 | mAP@.5:.95 | AP-small | P / R / F1 | Notes |
|------|--------|-------|-----|--------------------|--------|--------|-----------|----------|-----------|-------|
| _tbd_ | baseline-roboflow | Roboflow hosted | n/a | test | n/a | | | | | scored as-is via supervision.metrics |
| _tbd_ | rfdetr-medium-v1 | RFDETRMedium | 576 | | 50 | | | | | checkpoint_best_ema.pth |
| _tbd_ | yolo11-v1 | YOLO11 | | | | | | | | same-data baseline |

## End-to-end reading (CER/WER)

| Run ID | Grade | cell-level CER | text-level CER | WER | reading-order | Notes |
|--------|-------|----------------|----------------|-----|---------------|-------|
| _tbd_ | G1 | | | | predicted / oracle | liblouis back-translation |
