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


def ensure_portrait(bgr: np.ndarray, rotate_dir: str = "cw") -> np.ndarray:
    """All pages in our book are VERTICAL; some files were saved rotated (no EXIF tag),
    so landscape crops are turned upright.

    Direction matters: braille is orientation-sensitive - a cell rotated 180 degrees is a
    DIFFERENT character (dots 1<->6, 2<->5, 3<->4). CW and CCW both yield portrait but
    differ by 180, so the correct direction must be confirmed on real pages
    (see `--orient-preview`), not guessed.
    """
    h, w = bgr.shape[:2]
    if w <= h:
        return bgr
    code = cv2.ROTATE_90_CLOCKWISE if rotate_dir == "cw" else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(bgr, code)


def _covers_whole_frame(corners: np.ndarray, shape, tol: float = 0.97) -> bool:
    """A 'crop' whose quad is essentially the entire photo means detection failed
    silently (the classic symptom: every output has identical full-frame dimensions)."""
    h, w = shape[:2]
    quad_area = cv2.contourArea(corners.astype(np.float32))
    return quad_area >= tol * (h * w)


def crop(bgr: np.ndarray, do_scale: bool = True, portrait: bool = True,
         rotate_dir: str = "cw"):
    """-> (cropped_bgr, corners_or_None, method).

    IMPORTANT: run this on the RAW photo, before enhancement. Flat-field
    normalization deliberately flattens the page-vs-background illumination
    difference, which is exactly the signal page detection relies on - cropping
    enhanced images made ~25% of pages fail to full-frame.
    """
    def _finish(img, corners, method):
        if portrait:
            img = ensure_portrait(img, rotate_dir)
        return (normalize_scale(img) if do_scale else img), corners, method

    if _DOCALIGNER is not None:
        try:
            poly = _DOCALIGNER(bgr)
            pts = np.asarray(getattr(poly, "points", poly), dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] == 4:
                corners = order_corners(pts)
                if not _covers_whole_frame(corners, bgr.shape):
                    return _finish(warp_to_page(bgr, corners), corners, "docaligner")
        except Exception:  # noqa: BLE001
            pass
    corners, method = find_page_corners(bgr)
    if corners is None:
        return _finish(bgr, None, f"FAILED: {method}")
    if _covers_whole_frame(corners, bgr.shape):
        return _finish(bgr, None, f"FAILED: quad covers whole frame ({method})")
    return _finish(warp_to_page(bgr, corners), corners, method)


def run(input_dir: Path, out_dir: Path = None, preview: int = 0,
        do_scale: bool = True, portrait: bool = True, rotate_dir: str = "cw",
        enhance_after: bool = True) -> None:
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
        out, corners, method = crop(img, do_scale=do_scale, portrait=portrait,
                                    rotate_dir=rotate_dir)
        if corners is None:
            n_fail += 1
        if enhance_after:
            # Enhance AFTER cropping: page detection needs the raw illumination
            # difference, and enhancing only the page avoids wasting the CLAHE
            # dynamic range on background clutter.
            from annotate.enhance import enhance as _enhance
            out = _enhance(out)
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
    ap.add_argument("--input", required=True,
                    help="folder of RAW photos (do NOT pre-enhance - see crop() docstring)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--no-scale", action="store_true",
                    help="skip resizing to the training cell scale")
    ap.add_argument("--no-portrait", action="store_true",
                    help="keep landscape crops as-is instead of rotating upright")
    ap.add_argument("--rotate-dir", default="cw", choices=["cw", "ccw"],
                    help="direction for landscape->portrait (prelabel --auto-orient "
                         "resolves the remaining 180-degree ambiguity)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="skip illumination enhancement of the cropped page")
    a = ap.parse_args()
    run(Path(a.input), a.out, a.preview, do_scale=not a.no_scale,
        portrait=not a.no_portrait, rotate_dir=a.rotate_dir,
        enhance_after=not a.no_enhance)
