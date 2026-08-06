"""Phase 2 - detect the page (largest quadrilateral) and crop it to a top-down view.

Our photos are white book pages on a dark background, so a threshold + contour
approach is reliable and needs no model. DocAligner is used when installed
(more robust on cluttered backgrounds); otherwise the OpenCV path runs.

NOTE on Google ML Kit Document Scanner: Android-only (Google Play Services), no
Python/server API - it cannot be used here. See RESEARCH_PLAN.md.

Scale normalization: our DSLR cells are ~85px wide vs ~28px in the base datasets.
Detectors are scale-sensitive, so the cropped page is resized so a cell lands near
TARGET_CELL_PX, putting our images in the same scale regime as the training data.

    python scripts/annotate/crop_page.py --input "Braille Dataset" --preview 6
    python scripts/annotate/crop_page.py --input "Braille Dataset"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402
from annotate.enhance import _imread, _imwrite  # noqa: E402

try:
    from docaligner import DocAligner  # optional
    _DOCALIGNER = DocAligner()
except Exception:  # noqa: BLE001
    _DOCALIGNER = None


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_page_corners(bgr: np.ndarray, work_px: int = 1000):
    """Return (corners_full_res 4x2, method) or (None, reason).

    Threshold-based: the page is much brighter than the background, so Otsu on a
    blurred grayscale separates them cleanly. We take the largest contour and try to
    reduce it to 4 corners; if the page is curved (bound book) approxPolyDP may not
    give exactly 4, so we fall back to the min-area rectangle.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, work_px / max(h, w))
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # close gaps, drop specks
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, "no contour"
    c = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if area_frac < 0.15:
        return None, f"largest region only {area_frac:.0%} of frame"

    peri = cv2.arcLength(c, True)
    quad, method = None, ""
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            quad, method = approx.reshape(4, 2).astype(np.float32), f"contour(eps={eps})"
            break
    if quad is None:                       # curved page - use the min-area rectangle
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        method = "minAreaRect"
    return order_corners(quad / scale), method


def warp_to_page(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    wA, wB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    hA, hB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    W, H = int(round(max(wA, wB))), int(round(max(hA, hB)))
    if W < 10 or H < 10:
        return bgr
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_AREA)


def normalize_scale(bgr: np.ndarray,
                    source_cell_px: float = None,
                    target_cell_px: float = None) -> np.ndarray:
    """Resize so a braille cell is about target_cell_px wide (match training scale)."""
    source_cell_px = source_cell_px or config.APPROX_SOURCE_CELL_PX
    target_cell_px = target_cell_px or config.TARGET_CELL_PX
    f = float(target_cell_px) / float(source_cell_px)
    if abs(f - 1.0) < 0.02:
        return bgr
    h, w = bgr.shape[:2]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (max(1, int(w * f)), max(1, int(h * f))), interpolation=interp)


def crop(bgr: np.ndarray, do_scale: bool = True):
    """-> (cropped_bgr, corners_or_None, method)."""
    if _DOCALIGNER is not None:
        try:
            poly = _DOCALIGNER(bgr)
            pts = np.asarray(getattr(poly, "points", poly), dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] == 4:
                corners = order_corners(pts)
                out = warp_to_page(bgr, corners)
                return (normalize_scale(out) if do_scale else out), corners, "docaligner"
        except Exception:  # noqa: BLE001
            pass
    corners, method = find_page_corners(bgr)
    if corners is None:
        return (normalize_scale(bgr) if do_scale else bgr), None, f"FAILED: {method}"
    out = warp_to_page(bgr, corners)
    return (normalize_scale(out) if do_scale else out), corners, method


def run(input_dir: Path, out_dir: Path = None, preview: int = 0,
        do_scale: bool = True) -> None:
    out_dir = Path(out_dir or config.CONTRIB_CROPPED)
    paths = sorted(p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in config.IMAGE_EXTS)
    if not paths:
        print(f"No images in {input_dir}")
        return
    if preview:
        step = max(1, len(paths) // preview)
        paths = paths[::step][:preview]
        pv = Path(config.CONTRIB_PREVIEW) / "crop"
        pv.mkdir(parents=True, exist_ok=True)
        print(f"PREVIEW mode: {len(paths)} images -> {pv}")

    n_fail = 0
    for p in paths:
        img = _imread(p)
        out, corners, method = crop(img, do_scale=do_scale)
        if corners is None:
            n_fail += 1
        h, w = img.shape[:2]
        oh, ow = out.shape[:2]
        print(f"  {p.name}: {w}x{h} -> {ow}x{oh}   [{method}]")
        if preview:
            vis = img.copy()
            if corners is not None:
                cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 12)
            vis = cv2.resize(vis, (800, int(800 * h / w)), interpolation=cv2.INTER_AREA)
            outv = cv2.resize(out, (800, max(1, int(800 * oh / ow))),
                              interpolation=cv2.INTER_AREA)
            H = max(vis.shape[0], outv.shape[0])
            canvas = np.zeros((H, 1600, 3), np.uint8)
            canvas[:vis.shape[0], :800] = vis
            canvas[:outv.shape[0], 800:] = outv
            cv2.putText(canvas, f"detected [{method}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(canvas, f"cropped {ow}x{oh}", (810, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            _imwrite(Path(config.CONTRIB_PREVIEW) / "crop" / f"{p.stem}_crop.jpg", canvas)
        else:
            _imwrite(out_dir / p.name, out)
    print(f"\nDone. {len(paths) - n_fail}/{len(paths)} pages detected"
          f"{f' ({n_fail} fell back to full frame)' if n_fail else ''}.")
    print("DocAligner:", "in use" if _DOCALIGNER else "not installed (OpenCV path)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--no-scale", action="store_true",
                    help="skip resizing to the training cell scale")
    a = ap.parse_args()
    run(Path(a.input), a.out, a.preview, do_scale=not a.no_scale)
