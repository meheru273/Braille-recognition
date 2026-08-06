# Braille Recognition — Research & Dataset Plan

> Goal: build a **contributed braille object-detection dataset** (from your own page photos,
> semi-automatically labeled), then **train our own open RF-DETR detector** to replace the
> opaque Roboflow-hosted model, evaluate it rigorously, and **publish** the model (Hugging Face),
> the dataset (GitHub + Zenodo DOI), and a **paper**. The current HF Gradio Space stays as-is for now.

This plan is grounded in a verified research pass (6 topic agents + 3 adversarial fact-checkers).
Items marked ⚠️ were corrected or flagged uncertain by the fact-checkers — do not build on the
un-corrected version.

---

## 0. End-state (what "done" looks like)

- `braille-dataset/` — images + `_annotations.coco.json` (train/valid/test), page-level splits, datasheet, CC-BY-4.0, Zenodo DOI.
- `braille-rfdetr` — fine-tuned RF-DETR weights (safetensors) on Hugging Face, Apache-2.0, with a model card + metrics.
- A reproducible training/eval repo on GitHub (this repo, extended).
- A paper (ASSETS / W4A / ICCHP + arXiv preprint) whose centerpiece is **our open RF-DETR vs the hidden Roboflow baseline**.

---

## Key up-front decisions (recommended defaults in **bold**)

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | **Training compute** | Local GPU (≥16 GB VRAM) / Google Colab / cloud (Lambda, RunPod, etc.) | ✅ **FROZEN: local NVIDIA GPU ≥16 GB.** Train locally: RFDETRMedium@576 with `batch_size=4, grad_accum_steps=4` (raise if VRAM allows); try RFDETRLarge@704 if it fits. |
| D2 | **Label granularity** | (a) 26 letters a–z (matches current model), (b) **63-class six-dot cell encoding** (Angelina-style, language-agnostic), (c) single `braille-cell` class + separate classifier | ✅ **FROZEN: (b) 63-class six-dot encoding.** Pre-annotate with the existing a–z model, then map a–z → its fixed 6-dot pattern; draw manual boxes for non-letter cells. Canonical label = the 6-dot pattern (Unicode U+2800 range). |
| D3 | **Braille scope (v1)** | Grade-1 English letters / add digits+punctuation / Grade-2 contracted | **Grade-1 English (a–z + space)** for v1; expand later. Record grade+language in the datasheet (critical — a detector doesn't generalize across grades/languages). |
| D4 | **RF-DETR variant** | Nano/Small/**Medium (576)**/Large (704) | **Medium** to start (good small-object resolution on 16 GB); try **Large** if VRAM allows. Stay within Nano–Large (Apache-2.0). ⚠️ **Avoid XL/2XL — those are PML-1.0, not open.** |
| D5 | **Dataset redistribution** | Re-host Angelina/DSBI images / release only our photos + pointer scripts | **Release only OUR annotated photos + conversion scripts** and link to the originals, unless the authors grant redistribution. ⚠️ Neither Angelina nor DSBI ships a LICENSE — email the authors before re-hosting. |

---

## The verified stack (what we'll use)

| Step | Tool | License | Note |
|------|------|---------|------|
| Page auto-crop (primary) | **DocAligner** (`pip install docaligner-docsaid`) | Apache-2.0 | Deep 4-corner detector → OpenCV perspective warp. Runs CPU/GPU via ONNXRuntime. |
| Page auto-crop (fallback) | OpenCV Canny→`findContours`→`approxPolyDP(4)`→`getPerspectiveTransform` | Apache-2.0 | No weights, deterministic, easy to describe in the paper. |
| Dewarp (only if curved pages) | UVDoc | MIT | ⚠️ Now pip-importable via `transformers` (`PaddlePaddle/UVDoc_safetensors`), not only GitHub clone. Flat sheets usually don't need it. |
| ❌ Excluded | Google ML Kit Document Scanner | proprietary | ⚠️ **Android-only (Google Play Services). Cannot run in Python / HF Spaces.** Do not attempt. |
| ❌ Wrong tool | docTR | Apache-2.0 | It's a *text-region* OCR detector, not a *page* cropper. |
| Detections / format convert / save | **supervision** (`pip install supervision`) | Apache-2.0 | `sv.Detections`, `sv.DetectionDataset.as_coco()/as_yolo()`. Also gives mAP/confusion-matrix later. |
| Manual correction (dense boxes) | **CVAT** (self-host or app.cvat.ai) | MIT | Imports pre-labels as **editable** shapes (Label Studio makes them read-only until copied). |
| QA / triage / round-trip | FiftyOne | Apache-2.0 | Sort by confidence, find FP/FN, `dataset.annotate(backend="cvat")`, later `evaluate_detections()`. |
| Detector | **RF-DETR** (`pip install rfdetr`, Python ≥3.10) | Apache-2.0 (Nano–Large) | COCO format; resolution divisible by 56; `.train()/.predict()/.export()`. |
| Baseline metrics | supervision.metrics | MIT | Score the current Roboflow model with almost no glue. |
| Per-epoch val metrics | torchmetrics `MeanAveragePrecision(class_metrics=True)` | Apache-2.0 | Inside the training loop. |
| Final reported metrics | pycocotools `COCOeval` | BSD | ⚠️ raise `params.maxDets` (e.g. `[100,300,1000]`) — dense pages have hundreds of cells or AR is under-reported. |
| Split | scikit-learn `GroupShuffleSplit`/`StratifiedGroupKFold` | BSD | **Split by PAGE, never by cell** (leakage). |
| End-to-end reading eval | liblouis (back-translate) + jiwer (CER/WER) | LGPL / Apache-2.0 | Report Grade-1 vs Grade-2 separately. |

---

## Phase 0 — Setup & scaffolding  *(~0.5 day)*

1. ✅ Done. `scripts/` workspace (copyable to the remote GPU): `scripts/common/` (config + `braille.py` 63-class encoding), `scripts/dataset/` (Phase 1), `scripts/annotate/` (Phase 2), `scripts/model/`, `scripts/inference/`, `scripts/eval/`. HF Space files at repo root untouched.
2. ✅ Done. `scripts/requirements.txt` (separate from the Space's root `requirements.txt`).
3. ✅ Done. D1–D5 frozen in `scripts/README.md`.
4. ✅ Done. `scripts/EXPERIMENTS.md` log. RF-DETR logging extra is `rfdetr[loggers]` (see ⚠️).

**Deliverable:** repo scaffold + frozen config.

---

## Phase 1 — Acquire the base datasets  *(~1–2 days)*

1. Clone **AngelinaDataset** (`github.com/IlyaOvodov/AngelinaDataset`) — ~334 page photos, per-character boxes labeled 1–63 (six-dot encoding), incl. negatives + a user-photo test set. This is the **standard base** (it's the dataset behind Ovodov's ICCVW 2021 object-detection braille reader — the same photo→detector→characters design you're building).
2. Clone **DSBI** (`github.com/yeluo1994/DSBI`) — 114 double-sided images, per-cell recto+verso dot-state labels + skew/grid metadata. Combine for double-sided coverage.
3. Write converters → unified **COCO**: `angelina_to_coco.py` (LabelMe JSON / CSV → COCO) and `dsbi_to_coco.py` (parse the 8-number-per-cell `.txt` → boxes → COCO). Decide recto-only vs recto+verso class.
4. (Optional augmentation) Pull **Roboflow Universe "Braille Detection V2"** (~1,324 imgs, CC-BY-4.0) via the Roboflow SDK — vet labels, use as extra data only.
5. ⚠️ **Do NOT** use the Kaggle `shanks0465` set (28×28 single-char classification crops — no pages, no boxes; it's the classic trap).
6. ⚠️ Email Ilya Ovodov and the DSBI authors for redistribution permission; cite both papers regardless (see D5).

**Deliverable:** `base_coco/` with Angelina + DSBI in one COCO schema, plus license notes.

---

## Phase 2 — Build the contribution dataset: auto-crop → pre-annotate → QA  *(~1–2 weeks, human-in-loop)*

This is the pipeline you described. Four scripted stages + human review.

> **Strategy pivot (2026-08-07):** the hosted Roboflow model 507s on its serverless tier, and the
> replacement hosted workflow is 26-class (a–z) only. So the order is now: **Phase 3 split + Phase 4
> RF-DETR training on the Angelina+DSBI base set come FIRST**, and that 63-class model becomes the
> pre-annotator for our own photos (`prelabel.py --backend rfdetr`). The Roboflow backends remain as
> fallbacks. This is also the cleaner paper story: our own open model bootstraps our dataset.

0. **Enhance** (`scripts/annotate/enhance.py`): illumination-normalize each photo (white-balance + CLAHE + flat-field). ⚠️ **No deshadowing/binarization** — embossed braille is read from dot micro-shadows; removing them hurts detection. A/B enhance-on/off as an ablation.
1. **Auto-crop** (`scripts/annotate/crop_page.py`): for each photo, run DocAligner → 4 corners → `cv2.getPerspectiveTransform`+`warpPerspective` → save top-down page. Fall back to the OpenCV contour method if DocAligner returns no valid quad.
2. **Pre-annotate** (`scripts/annotate/prelabel.py`, **two swappable backends**): `--backend roboflow` (existing hosted model — bootstrap the first labels) or `--backend rfdetr --weights <ckpt>` (our trained model, Phase 4+). Convert predictions → `sv.Detections` → write **COCO** (canonical) + per-image **YOLO** `.txt`. ⚠️ Low accept threshold (~0.25–0.35) + class-agnostic NMS @IoU~0.5 — on dense pages deleting a false box is cheaper than drawing a missed one.
3. **Human correction** (CVAT): upload COCO/YOLO pre-labels → they appear as editable boxes → fix/relabel/delete. Use FiftyOne to sort by confidence and hunt FP/FN first.
4. **Cross-check / QA:** boxes-per-row regularity check (braille is a tight grid), spot-check a %, log inter-annotator notes for the paper. Map a–z → 6-dot class here if using D2(b).

**Deliverable:** your photos → corrected COCO annotations = the **contribution dataset**.

---

## Phase 3 — Consolidate, define classes, split  *(~1–2 days)*

1. Merge Angelina + DSBI + your contribution set into one COCO dataset with the **final class schema** (D2/D3). Normalize category ids across sources.
2. De-duplicate near-identical photos.
3. **Split by PAGE** with `GroupShuffleSplit` / `StratifiedGroupKFold` (groups = page/document id). ⚠️ Never let cells from the same page span train/test. Small data → use GroupKFold and report mean±std.
4. Emit the RF-DETR layout: `dataset/{train,valid,test}/` each with images + `_annotations.coco.json`.

**Deliverable:** final training-ready COCO dataset + frozen splits (committed).

---

## Phase 4 — Train & fine-tune RF-DETR (+ a YOLO baseline)  *(~3–5 days incl. runs)*

1. `pip install rfdetr` (Python ≥3.10, CUDA GPU). ⚠️ For logging use the **`rfdetr[loggers]`** extra — **there is no `rfdetr[metrics]` extra** (fact-check correction).
2. Fine-tune from COCO-pretrained weights:
   ```python
   from rfdetr import RFDETRMedium
   model = RFDETRMedium()                     # loads COCO-pretrained weights
   model.train(dataset_dir="dataset", epochs=50,
               batch_size=4, grad_accum_steps=4,   # effective batch 16
               lr=1e-4, output_dir="output")
   ```
   Resolution must be **divisible by 56**; train at the highest your VRAM allows (small braille dots need resolution). Report from `checkpoint_best_ema.pth`.
3. Train a **YOLO (v11/v12)** baseline on the **identical** COCO data — independent studies show YOLO can match/beat RF-DETR on the very smallest objects / mAP@50:95, so a same-data head-to-head is the credible evidence for the paper.
4. Keep the **hidden Roboflow workflow** as a third comparison point (scored as-is, no retraining).

**Deliverable:** fine-tuned RF-DETR checkpoint(s) + YOLO baseline + training logs.

---

## Phase 5 — Evaluation  *(~2–3 days)*

Two layers, strict page-level test set:

1. **Detection:** headline **mAP@0.5:0.95** + **mAP@0.5**, plus AP-small, precision/recall/F1 at the val-chosen F1-optimal confidence, **per-class AP**, PR curves, confusion matrix.
   - Score the **Roboflow baseline** with `supervision.metrics` (minimal glue).
   - Per-epoch RF-DETR val with `torchmetrics MeanAveragePrecision(class_metrics=True)`.
   - **Final** reported test number with **pycocotools** (⚠️ raise `maxDets`).
2. **End-to-end reading:** assemble cells → reading order (upgrade `organize_text_by_rows` to a robust grid fit) → back-translate with **liblouis** → **CER/WER** via **jiwer**, reported **Grade-1 vs Grade-2 separately**. Ablate reading-order with oracle ordering to isolate assembly error from detection error.
3. **Ablations** (one factor at a time): page-crop on/off, input resolution, pretrained vs scratch, augmentation, train-set-size learning curve.

**Deliverable:** results tables + figures (baseline vs RF-DETR vs YOLO), CER/WER tables.

---

## Phase 6 — Package & publish  *(~1–2 weeks, overlaps writing)*

1. **Model → Hugging Face:** dedicated repo `hf.co/<you>/braille-rfdetr`, weights as `.safetensors` via git-LFS, full model card (YAML `pipeline_tag: object-detection`, `license: apache-2.0`, `datasets:` link, metrics, **limitations + accessibility ethics**). Point the existing Gradio Space at these weights via `hf_hub_download` (keep the Space otherwise as-is).
2. **Dataset:** **Zenodo** as the citable source of record (DOI via GitHub-release integration, **CC-BY-4.0**), **mirrored** to HF Datasets (viewer + `load_dataset`) and optionally Roboflow Universe. Write a **datasheet**: braille grade+language, provenance/consent, "assistive aid, not certified transcription."
3. **Code:** this GitHub repo — crop + pre-annotate + train + eval scripts, `CITATION.cff`, `.zenodo.json`, environment pins.
4. **Paper:** target **ACM ASSETS'26** (Porto, Oct 25–28 2026; technical-paper deadline **~Apr 22 2026 AoE**; ⚠️ **accessible/tagged PDF is mandatory or desk-reject**). Alternatives: **W4A** (rewards released code+data; has an Accessibility Challenge track that fits the Gradio demo) and **ICCHP 2026** (Springer LNCS — ⚠️ verify 2026 dates from the Call PDF). Post an **arXiv preprint** regardless. Cite RF-DETR (arXiv 2511.09554, ICLR 2026 — ⚠️ its headline contribution is *Neural Architecture Search*, not just the DINOv2 backbone swap).

**Deliverable:** public model + dataset DOI + preprint + submitted paper.

---

## Consolidated task checklist

- [x] **P0.1** Repo scaffold `scripts/*` + pinned `scripts/requirements.txt` + `braille.py` 63-class util
- [x] **P0.2** Freeze decisions D1–D5 in `scripts/README.md`
- [x] **P1.0** `dataset_downloader.py` + `annotator.py` + `orchestrator.py` (download→inspect→annotate)
- [ ] **P1.1** Clone + inspect AngelinaDataset; write `angelina_to_coco.py`
- [ ] **P1.2** Clone + inspect DSBI; write `dsbi_to_coco.py`
- [ ] **P1.3** (opt) Pull + vet Roboflow "Braille Detection V2" (CC-BY-4.0)
- [ ] **P1.4** Email Angelina/DSBI authors re: redistribution; add citations
- [ ] **P2.1** `crop_page.py` (DocAligner + OpenCV fallback)
- [ ] **P2.2** `prelabel.py` (existing detector → supervision → COCO + YOLO)
- [ ] **P2.3** Stand up CVAT + FiftyOne; correct pre-labels
- [ ] **P2.4** QA (grid regularity, spot-check, a–z→6-dot mapping)
- [ ] **P3.1** Merge sources → unified COCO; normalize classes; de-dup
- [ ] **P3.2** Page-level split (`GroupShuffleSplit`/`GroupKFold`); freeze splits
- [ ] **P4.1** Install rfdetr; fine-tune RFDETRMedium (then Large)
- [ ] **P4.2** Train YOLO baseline on identical data
- [ ] **P5.1** Score Roboflow baseline (supervision.metrics)
- [ ] **P5.2** Detection metrics (pycocotools final; per-class AP; confusion matrix; PR)
- [ ] **P5.3** End-to-end CER/WER (liblouis + jiwer, Grade-1/2)
- [ ] **P5.4** Ablations + learning curve
- [ ] **P6.1** Publish model + card to HF; wire Space to weights
- [ ] **P6.2** Publish dataset (Zenodo DOI + HF Datasets mirror) + datasheet
- [ ] **P6.3** Release code (CITATION.cff, .zenodo.json)
- [ ] **P6.4** Write + submit paper (arXiv preprint; ASSETS/W4A/ICCHP)

---

## Risks & fact-check corrections (don't build on the wrong version)

- ⚠️ **`rfdetr[metrics]` does not exist** → use **`rfdetr[loggers]`** for TensorBoard/W&B. (`.train(tensorboard=True, wandb=True)` flags are real.)
- ⚠️ **Google ML Kit Document Scanner is Android-only** — excluded; use DocAligner + OpenCV.
- ⚠️ **Angelina & DSBI have no LICENSE files** — redistribution needs author permission; release your own photos + pointers instead (D5).
- ⚠️ **Split by page, not cell** — the #1 way detection papers get rejected for leakage.
- ⚠️ **pycocotools default `maxDets=100`** under-reports recall on dense braille pages — raise it.
- ⚠️ **Kaggle `shanks0465`** and most "braille datasets" are classification crops — useless for detection.
- ⚠️ RF-DETR **XL/2XL are PML-1.0 (not open)** — stay within **Nano–Large (Apache-2.0)** for a freely publishable model.
- Uncertain: exact ICCHP 2026 dates/limits; the greenfruit RF-DETR-vs-YOLO numbers (arXiv 2504.13099) — verify before quoting; DSBI's exact redistribution wording (ACM page returned 403).
- Small personal dataset → single-split mAP is noisy; use GroupKFold and report mean±std.

## Key references

- Angelina: github.com/IlyaOvodov/AngelinaDataset · Ovodov ICCVW 2021 (arXiv:2012.12412)
- DSBI: github.com/yeluo1994/DSBI · arXiv:1811.10893
- RF-DETR: github.com/roboflow/rf-detr · rfdetr.roboflow.com/learn/train · arXiv:2511.09554 (ICLR 2026)
- DocAligner: pypi.org/project/docaligner-docsaid · supervision: supervision.roboflow.com
- Splitting: scikit-learn GroupShuffleSplit · Eval: pycocotools, torchmetrics detection · liblouis + jiwer
- Publish: HF model-release-checklist · Zenodo GitHub integration · ASSETS'26 assets26.sigaccess.org
