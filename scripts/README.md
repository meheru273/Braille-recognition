# `scripts/` — Braille dataset + model pipeline

Self-contained working area for the research effort (full plan: [`../RESEARCH_PLAN.md`](../RESEARCH_PLAN.md)).
Copy this whole folder to the remote GPU (AnyDesk) and it runs there — data is regenerated
by re-running the downloader, so nothing large needs copying.

The Hugging Face Gradio Space (repo-root `app.py`, `detector.py`, `assistant.py`) is untouched.

## Layout

```
scripts/
  common/       config.py (paths, dataset URLs), braille.py (63-class six-dot encoding)
  dataset/      Phase 1 — dataset_downloader.py, annotator.py, orchestrator.py
  annotate/     Phase 2 — enhance + crop + pre-annotate OUR photos (see annotate/README.md)
  model/        Phase 4 — RF-DETR + YOLO training
  inference/    run a trained model on new images
  eval/         Phase 5 — metrics + CER/WER
  data/         (gitignored) raw/ cloned datasets, coco/ unified output
  requirements.txt   EXPERIMENTS.md
```

## Frozen decisions

| Key | Decision |
|-----|----------|
| Compute (D1) | Local NVIDIA GPU ≥16 GB. RFDETRMedium@576, `batch_size=4, grad_accum_steps=4`. |
| Labels (D2) | 63-class six-dot encoding (ids 1–63 = dot bitmask). See `common/braille.py`. |
| Scope v1 (D3) | Grade-1 English (a–z + space). |
| RF-DETR (D4) | Nano–Large only (Apache-2.0). Never XL/2XL. |
| Redistribution (D5) | Release our photos + scripts + pointers; contact Angelina/DSBI authors before re-hosting. |

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux
pip install -r scripts/requirements.txt
python scripts/common/braille.py  # sanity check the a-z encoding table
```

## Phase 1 — build the base dataset

```bash
python scripts/dataset/orchestrator.py --steps download            # clone Angelina + DSBI
python scripts/dataset/orchestrator.py --steps annotate            # -> unified COCO
python scripts/dataset/orchestrator.py --steps visualize           # spot-check boxes
# output: scripts/data/coco/_annotations.coco.json + images/  (+ _viz/ previews)
```

`--steps inspect` prints each dataset's real on-disk structure (useful if a source changes).

### Confirmed source formats

| Dataset | Annotation used | Geometry | Label |
|---|---|---|---|
| **Angelina** | `X.labeled.csv` | `left;top;right;bottom` **normalized [0,1]** | integer **class id 1–63**, same bit convention as ours |
| **DSBI** | `X+recto.txt` | grid: line 1 = x-centers (2/cell col), line 2 = y-centers (3/cell row) | per cell `row col d1..d6` flags → dot mask |

Both parsers are verified: Angelina CSV→pixels matches the paired LabelMe JSON to <1 px, and
DSBI grid→boxes was confirmed visually against a rendered page. Angelina is **Russian** braille,
so it uses the full 63-symbol space (not just a–z) — which is why we chose the six-dot schema.
The `.json` (LabelMe) files label shapes with the *Cyrillic character*, so we use the CSV instead.

The orchestrator is extensible: new steps (enhance, crop, prelabel, split, train) register in
`STEPS` in `dataset/orchestrator.py`.
