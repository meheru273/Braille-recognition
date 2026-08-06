"""Phase 2 - illumination normalization for embossed-braille photos.

WHY NOT A GENERIC "SMART SCANNER":
Embossed braille is visible ONLY through the micro-shadow/highlight on each raised
dot. Document-scanner "magic colour"/deshadow/binarize modes remove exactly that
signal, so they make braille HARDER to detect, not easier. We therefore only
normalize ILLUMINATION (global gradients, colour cast) and boost LOCAL contrast,
which amplifies the dot shading instead of erasing it.

Stages (all optional, gentle by default):
  1. gray-world white balance      - removes colour cast
  2. flat-field / background division - removes global lighting gradients + soft shadows
  3. CLAHE on the LAB L channel    - boosts LOCAL contrast, i.e. the dot shading

    python scripts/annotate/enhance.py --input "Braille Dataset" --preview 6
    python scripts/annotate/enhance.py --input "Braille Dataset"        # process all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Gray-world: scale each channel so their means match."""
    result = bgr.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    for c in range(3):
        if means[c] > 1e-6:
            result[..., c] *= gray / means[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def flat_field(bgr: np.ndarray, sigma_frac: float = 0.06, strength: float = 1.0,
               work_px: int = 512) -> np.ndarray:
    """Divide by a heavily blurred copy to flatten lighting gradients and soft shadows.

    The blur radius is a fraction of the image size, so it is far larger than a braille
    dot: dot-scale detail survives while the slow illumination ramp is removed.
    strength in [0,1] blends between the original and the fully flattened result.

    The background is estimated on a `work_px` thumbnail and upscaled - a low-pass
    filter has no high-frequency content to lose, and this is ~100x faster than
    blurring a 24MP image with a 300px kernel.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, work_px / max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    k = int(max(small.shape[:2]) * sigma_frac) | 1          # odd kernel
    bg_small = cv2.GaussianBlur(small, (k, k), 0)
    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    background = np.maximum(background, 1.0)
    src = bgr.astype(np.float32)
    flat = src / background * background.mean(axis=(0, 1), keepdims=True)
    out = (1.0 - strength) * src + strength * flat
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_l(bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB - local contrast, so dot shading is amplified."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance(bgr: np.ndarray, *, do_wb: bool = True, do_flat: bool = True,
            do_clahe: bool = True, clip: float = 2.0, grid: int = 8,
            flat_strength: float = 1.0) -> np.ndarray:
    out = bgr
    if do_wb:
        out = white_balance(out)
    if do_flat:
        out = flat_field(out, strength=flat_strength)
    if do_clahe:
        out = clahe_l(out, clip=clip, grid=grid)
    return out


def _imread(path: Path) -> np.ndarray:
    # cv2.imread chokes on non-ASCII paths on Windows; go through numpy
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {path}")
    return img


def _imwrite(path: Path, img: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(str(path))


def _side_by_side(before: np.ndarray, after: np.ndarray, max_w: int = 1600) -> np.ndarray:
    h, w = before.shape[:2]
    scale = min(1.0, (max_w / 2) / w)
    size = (int(w * scale), int(h * scale))
    b = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    cv2.putText(b, "before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(a, "after", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return np.hstack([b, a])


def dot_contrast(bgr: np.ndarray) -> float:
    """Proxy for 'how visible are the dots': std-dev of a high-pass (dot-scale) filter.
    Higher = more dot detail. Used to sanity-check that enhancement helps."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo = cv2.GaussianBlur(gray, (0, 0), 6)
    return float((gray - lo).std())


def run(input_dir: Path, out_dir: Path = None, preview: int = 0, **kw) -> None:
    out_dir = Path(out_dir or config.CONTRIB_ENHANCED)
    paths = sorted(p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in config.IMAGE_EXTS)
    if not paths:
        print(f"No images in {input_dir}")
        return
    if preview:
        step = max(1, len(paths) // preview)
        paths = paths[::step][:preview]
        prev_dir = Path(config.CONTRIB_PREVIEW) / "enhance"
        prev_dir.mkdir(parents=True, exist_ok=True)
        print(f"PREVIEW mode: {len(paths)} images -> {prev_dir}")

    for p in paths:
        img = _imread(p)
        out = enhance(img, **kw)
        before_c, after_c = dot_contrast(img), dot_contrast(out)
        if preview:
            pv = Path(config.CONTRIB_PREVIEW) / "enhance"
            _imwrite(pv / f"{p.stem}_full.jpg", _side_by_side(img, out))
            # 1:1 centre crop - at page scale the dots vanish, so also show detail
            h, w = img.shape[:2]
            ch, cw = min(700, h), min(700, w)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            _imwrite(pv / f"{p.stem}_detail.jpg",
                     _side_by_side(img[y0:y0 + ch, x0:x0 + cw],
                                   out[y0:y0 + ch, x0:x0 + cw], max_w=1600))
        else:
            _imwrite(out_dir / p.name, out)
        arrow = "UP " if after_c > before_c else "down"
        print(f"  {p.name}: dot-contrast {before_c:.2f} -> {after_c:.2f}  ({arrow})")
    print(f"\nDone. {'Previews' if preview else 'Enhanced images'} in "
          f"{(Path(config.CONTRIB_PREVIEW) / 'enhance') if preview else out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of raw photos")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=0,
                    help="write N before/after comparisons instead of processing all")
    ap.add_argument("--no-wb", action="store_true")
    ap.add_argument("--no-flat", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--grid", type=int, default=8, help="CLAHE tile grid")
    ap.add_argument("--flat-strength", type=float, default=1.0)
    a = ap.parse_args()
    run(Path(a.input), a.out, a.preview, do_wb=not a.no_wb, do_flat=not a.no_flat,
        do_clahe=not a.no_clahe, clip=a.clip, grid=a.grid, flat_strength=a.flat_strength)
