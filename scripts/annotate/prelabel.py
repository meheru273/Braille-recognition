"""Phase 2 - model-assisted pre-annotation of OUR photos.

Runs a detector over each cropped page and writes COCO (canonical) + per-image YOLO
labels for human correction in CVAT. Two swappable backends, same input/output:

  --backend roboflow-local  our Roboflow model, weights run LOCALLY via the
                            `inference` package (pip install inference). Preferred:
                            no hosted memory limits, no per-call cost, GPU-friendly.
  --backend roboflow-http   the hosted serverless endpoint (inference_sdk).
                            NOTE 2026-08: this model returns 507 "model loading failed"
                            on the hosted tier regardless of input size - kept because
                            it may work on another plan/later.
  --backend rfdetr          our own fine-tuned RF-DETR checkpoint (Phase 4+).

TILING: braille cells are small and pages are large, so full-page inference both
overflows the hosted tier and detects poorly. sv.InferenceSlicer runs the detector on
overlapping tiles and merges results - the standard fix for dense small objects.

ORIENTATION: our pages are all vertical but 31/100 files were saved rotated with no
EXIF tag. Direction cannot be guessed - braille rotated 180 degrees is a DIFFERENT
character. With --auto-orient we run the detector on both 90-degree rotations and keep
the one the model scores higher; the model was trained on upright braille, so it
resolves this reliably. Geometric heuristics were tried and are not dependable on
curved (bound-book) pages.

    python scripts/annotate/prelabel.py --input scripts/data/contrib/cropped \\
        --backend roboflow-local --auto-orient
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import braille, config  # noqa: E402
from annotate.enhance import _imread, _imwrite  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(config.REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

import supervision as sv  # noqa: E402

# Pre-labelling is HUMAN-in-the-loop: propose generously and let the reviewer delete.
# Deleting a false box is far cheaper than drawing a missed one on a dense page.
DEFAULT_CONF = 0.30
DEFAULT_NMS_IOU = 0.50
TILE = 640
TILE_OVERLAP = 0.25


# --------------------------------------------------------------------------------------
# Backends: each returns a callable(np.ndarray BGR) -> sv.Detections
# --------------------------------------------------------------------------------------
def backend_roboflow_local(conf: float):
    from inference import get_model
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "braille-to-text-custom-kvzne/1")
    model = get_model(model_id=model_id, api_key=os.getenv("ROBOFLOW_API_KEY"))
    print(f"  backend: roboflow-local  model={model_id}")

    def run(image: np.ndarray) -> sv.Detections:
        res = model.infer(image, confidence=conf)[0]
        return sv.Detections.from_inference(res)
    return run


def backend_roboflow_http(conf: float):
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                                 api_key=os.getenv("ROBOFLOW_API_KEY"))
    ws = os.getenv("ROBOFLOW_WORKSPACE", "braille-image")
    wf = os.getenv("ROBOFLOW_WORKFLOW", "braille-to-text-custom-v1-logic")
    print(f"  backend: roboflow-http  workspace={ws} workflow={wf}")

    def run(image: np.ndarray) -> sv.Detections:
        tmp = Path(os.environ.get("TEMP", "/tmp")) / "_prelabel_tile.jpg"
        _imwrite(tmp, image)
        res = client.run_workflow(workspace_name=ws, workflow_id=wf,
                                  images={"image": str(tmp)}, use_cache=True)
        r0 = res[0] if isinstance(res, list) and res else res
        preds = r0.get("predictions", r0)
        if isinstance(preds, dict):
            preds = preds.get("predictions", [])
        return sv.Detections.from_inference({"predictions": preds,
                                             "image": {"width": image.shape[1],
                                                       "height": image.shape[0]}})
    return run


def backend_rfdetr(conf: float, weights: str, size: str = "medium"):
    import rfdetr
    cls = {"nano": "RFDETRNano", "small": "RFDETRSmall",
           "medium": "RFDETRMedium", "large": "RFDETRLarge"}[size]
    model = getattr(rfdetr, cls)(pretrain_weights=weights)
    print(f"  backend: rfdetr-{size}  weights={weights}")

    def run(image: np.ndarray) -> sv.Detections:
        return model.predict(image, threshold=conf)
    return run


def make_backend(name: str, conf: float, weights: str = None, size: str = "medium"):
    if name == "roboflow-local":
        return backend_roboflow_local(conf)
    if name == "roboflow-http":
        return backend_roboflow_http(conf)
    if name == "rfdetr":
        if not weights:
            raise SystemExit("--backend rfdetr needs --weights <checkpoint.pth>")
        return backend_rfdetr(conf, weights, size)
    raise SystemExit(f"unknown backend {name!r}")


# --------------------------------------------------------------------------------------
def sliced_detect(detect_fn, image: np.ndarray, conf: float, iou: float) -> sv.Detections:
    """Tiled inference + merge. Braille cells are tiny relative to a page."""
    slicer = sv.InferenceSlicer(
        callback=lambda tile: detect_fn(tile),
        slice_wh=(TILE, TILE),
        overlap_ratio_wh=(TILE_OVERLAP, TILE_OVERLAP),
        iou_threshold=iou,
    )
    det = slicer(image)
    return det.with_nms(threshold=iou, class_agnostic=True)


def score_orientation(detect_fn, image: np.ndarray, conf: float, iou: float):
    """Return (best_image, rotation_used, {rot: (n, mean_conf)}).
    The detector was trained on upright braille, so the rotation it scores highest is
    the correct one - this also settles the 180-degree ambiguity that geometry cannot."""
    scores, best, best_key, best_val = {}, image, "none", -1.0
    if image.shape[1] > image.shape[0]:
        cands = {"cw": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                 "ccw": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    else:
        cands = {"none": image, "180": cv2.rotate(image, cv2.ROTATE_180)}
    for key, im in cands.items():
        det = sliced_detect(detect_fn, im, conf, iou)
        n = len(det)
        mc = float(det.confidence.mean()) if n else 0.0
        scores[key] = (n, round(mc, 3))
        val = n * mc                      # count x quality
        if val > best_val:
            best, best_key, best_val = im, key, val
    return best, best_key, scores


def to_class_id(name) -> int:
    """Roboflow classes are Latin letters; our schema is the 63-class dot mask."""
    s = str(name).strip().lower()
    if s in braille.LETTER_TO_CLASS:
        return braille.LETTER_TO_CLASS[s]
    if s.isdigit() and 1 <= int(s) <= 63:
        return int(s)
    return braille.dots_string_to_class(s)


def run(input_dir: Path, backend: str = "roboflow-local", conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_NMS_IOU, auto_orient: bool = False, limit: int = 0,
        weights: str = None, out_dir: Path = None, model_size: str = "medium") -> None:
    out_dir = Path(out_dir or config.CONTRIB_PRELABEL)
    img_out, lbl_out = out_dir / "images", out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    prev = Path(config.CONTRIB_PREVIEW) / "prelabel"
    prev.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in config.IMAGE_EXTS)
    if limit:
        paths = paths[:limit]
    if not paths:
        print(f"No images in {input_dir}")
        return
    detect_fn = make_backend(backend, conf, weights, model_size)
    print(f"  {len(paths)} images | conf>={conf} nms_iou={iou} tile={TILE}")

    images, annotations, ann_id = [], [], 0
    for idx, p in enumerate(paths, 1):
        img = _imread(p)
        if auto_orient:
            img, rot, scores = score_orientation(detect_fn, img, conf, iou)
            det = sliced_detect(detect_fn, img, conf, iou)
            note = f"orient={rot} {scores}"
        else:
            det = sliced_detect(detect_fn, img, conf, iou)
            note = ""
        h, w = img.shape[:2]
        _imwrite(img_out / p.name, img)
        images.append({"id": idx, "width": w, "height": h,
                       "file_name": p.name, "source": "contrib"})
        lines = []
        names = det.data.get("class_name", [None] * len(det))
        for (x1, y1, x2, y2), cls_name, cf in zip(det.xyxy, names, det.confidence):
            try:
                cid = to_class_id(cls_name)
            except ValueError:
                continue
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": idx, "category_id": cid,
                                "bbox": [float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1)],
                                "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
                                "score": float(cf)})
            lines.append(f"{cid-1} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (lbl_out / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if idx <= 6:
            vis = img.copy()
            for (x1, y1, x2, y2) in det.xyxy:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            _imwrite(prev / f"{p.stem}_pred.jpg", vis)
        print(f"  [{idx}/{len(paths)}] {p.name}: {len(det)} boxes {note}")

    coco = {"info": {"description": "Braille contribution set - MODEL PRE-LABELS "
                                    "(needs human correction)"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": braille.coco_categories()}
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"\nWrote {out_dir/'_annotations.coco.json'}  "
          f"({len(images)} images, {len(annotations)} boxes)")
    print("NEXT: import into CVAT (pre-labels are editable there), correct, then export COCO.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.CONTRIB_CROPPED))
    ap.add_argument("--backend", default="roboflow-local",
                    choices=["roboflow-local", "roboflow-http", "rfdetr"])
    ap.add_argument("--weights", default=None, help="rfdetr checkpoint")
    ap.add_argument("--model-size", default="medium",
                    choices=["nano", "small", "medium", "large"])
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_NMS_IOU)
    ap.add_argument("--auto-orient", action="store_true",
                    help="let the detector decide the page rotation")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(Path(a.input), a.backend, a.conf, a.iou, a.auto_orient, a.limit, a.weights,
        model_size=a.model_size)
