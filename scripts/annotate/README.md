# `annotate/` — Phase 2: enhance → crop → pre-annotate OUR photos

Turns your raw braille page photos into corrected COCO labels = the **contribution dataset**.
Pipeline (each becomes an orchestrator step): **enhance → crop → pre-annotate → (human QA)**.

## 1. `enhance.py` — illumination normalization (NOT deshadowing)

Your photos vary in lighting; we even that out **before** detection. Planned stages, gentle by default:

- White balance (gray-world / simple) + `cv2.createCLAHE` on the L channel (LAB) for local contrast.
- Illumination normalization by dividing the image by a heavily-blurred copy of itself
  (flat-field / background division) to remove global lighting gradients.

⚠️ **Braille-specific caveat — the reason we do NOT use a generic "ML-Kit-style smart scanner":**
embossed braille is detected from the **micro-shadows/highlights of the raised dots**. Binarization
and shadow-removal ("magic color") erase exactly that cue and will *hurt* detection. So we normalize
global illumination only and keep local dot shadows intact. Every enhancement must be validated by
re-running detection and checking mAP does not drop (add an ablation: enhance on/off).

Options considered (choose per results):
- **CLAHE + flat-field (OpenCV)** — default, fast, preserves dot shadows, fully explainable.
- **DocRes** (CVPR 2024, generalist document restoration: appearance-enhancement / deshadowing) —
  the closest "smart enhancer"; use its *appearance* mode cautiously and A/B it against no-enhance,
  because its deshadowing may remove dot cues.
- **Retinex / MSRCP** — classic illumination correction, middle ground.

## 2. `crop_page.py` — page auto-crop (largest quadrilateral)

- Primary: **DocAligner** (`docaligner-docsaid`, Apache-2.0) → 4 corners → `cv2.getPerspectiveTransform` + `warpPerspective`.
- Fallback: classic OpenCV `Canny → findContours → approxPolyDP(4) → order → warp` (no weights).
- ❌ Not Google ML Kit Document Scanner — Android-only, cannot run in Python/server.

## 3. `prelabel.py` — model-assisted pre-annotation, **two swappable backends**

You asked for both. Same input/output; pick via `--backend`:
- `--backend roboflow` — the existing hosted Roboflow model (works well today; reuses the Space's
  `detector.py` / inference_sdk). **Use this to bootstrap the first labels.**
- `--backend rfdetr --weights <ckpt>` — our own trained RF-DETR, once it exists (Phase 4). Lets us
  re-label with the improved model and compare.

Both convert predictions → `supervision.Detections` → write **COCO** (canonical) + per-image **YOLO**
`.txt` next to each image. Use a **low** accept threshold (~0.25–0.35) + class-agnostic NMS @IoU~0.5 —
on dense pages deleting a false box is cheaper than drawing a missed one. Map a–z → 6-dot class here.

## 4. Human QA

Correct in **CVAT** (imports pre-labels as editable boxes); triage with **FiftyOne** (sort by
confidence, find FP/FN). Grid-regularity check per page as an automated sanity signal.
