"""Phase 3b - window the split dataset into fixed-size tiles for training.

WHY (measured, see git history): RF-DETR resizes whole images to its training
resolution (576), so on our page-sized images cells shrink to ~11-12px effective -
while inference runs on 640px tiles where cells appear at ~25px. The model learned
"cell ~= 12px" and then saw dots (~9px) at inference, boxing single dots. Pages also
carry up to ~620 cells, starving DETR's ~300-query budget (val recall capped ~0.5).
Tiling the TRAINING data to the same 640px geometry fixes both: cells train at the
inference scale and each tile holds ~30-60 boxes.

Leakage: tiles inherit their source page's split (we tile train/valid/test
separately), so the page-level group guarantee is preserved.

    python scripts/dataset/tile.py            # dataset/ -> dataset_tiled/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402

try:
    from PIL import Image
except ImportError:
    Image = None

TILE = 640
OVERLAP = 0.2            # 128px >> cell size (~28px): every cell is whole in >=1 tile
KEEP_FRACTION = 0.8      # box kept if >=80% of its area lies inside the tile
EMPTY_KEEP_RATE = 8      # keep 1 in N empty tiles as background/negatives


def _starts(full: int, tile: int, step: int):
    """Window start offsets covering [0, full) with the final tile flush to the edge."""
    if full <= tile:
        return [0]
    xs = list(range(0, full - tile, step))
    xs.append(full - tile)
    return xs


def tile_split(split_dir: Path, out_dir: Path) -> tuple:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(TILE * (1 - OVERLAP))
    images, annotations = [], []
    img_id = ann_id = 0
    n_empty_kept = n_empty_dropped = 0

    for im in coco["images"]:
        src = split_dir / im["file_name"]
        if not src.exists():
            continue
        page = Image.open(src)
        W, H = page.size
        boxes = anns_by_img[im["id"]]
        for ty in _starts(H, TILE, step):
            for tx in _starts(W, TILE, step):
                tw, th = min(TILE, W - tx), min(TILE, H - ty)
                kept = []
                for a in boxes:
                    x, y, w, h = a["bbox"]
                    ix1, iy1 = max(x, tx), max(y, ty)
                    ix2, iy2 = min(x + w, tx + tw), min(y + h, ty + th)
                    iw, ih = ix2 - ix1, iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue
                    if (iw * ih) / (w * h) < KEEP_FRACTION:
                        continue      # cell mostly outside; a neighbouring tile owns it
                    kept.append((a["category_id"], [ix1 - tx, iy1 - ty, iw, ih]))
                if not kept:
                    # deterministic thinning of background tiles (reproducible builds)
                    key = f"{im['file_name']}:{tx}:{ty}".encode()
                    if int(hashlib.md5(key).hexdigest(), 16) % EMPTY_KEEP_RATE:
                        n_empty_dropped += 1
                        continue
                    n_empty_kept += 1
                img_id += 1
                name = f"{Path(im['file_name']).stem}__t{tx}_{ty}.jpg"
                page.crop((tx, ty, tx + tw, ty + th)).save(out_dir / name, quality=92)
                images.append({"id": img_id, "width": tw, "height": th,
                               "file_name": name, "source": im.get("source", "?"),
                               "group": im.get("group", im["file_name"]),
                               "page": im["file_name"]})
                for cid, bbox in kept:
                    ann_id += 1
                    annotations.append({"id": ann_id, "image_id": img_id,
                                        "category_id": cid, "bbox": [float(v) for v in bbox],
                                        "area": float(bbox[2] * bbox[3]), "iscrowd": 0})

    out = {"info": {"description": coco.get("info", {}).get("description", "") +
                    f" [tiled {TILE}px overlap {OVERLAP}]"},
           "licenses": coco.get("licenses", []), "images": images,
           "annotations": annotations, "categories": coco["categories"]}
    (out_dir / "_annotations.coco.json").write_text(json.dumps(out), encoding="utf-8")
    return len(images), len(annotations), n_empty_kept, n_empty_dropped


def run(dataset_dir: Path = None, out_root: Path = None) -> None:
    if Image is None:
        raise SystemExit("Pillow required")
    dataset_dir = Path(dataset_dir or (config.DATA_DIR / "dataset"))
    out_root = Path(out_root or (config.DATA_DIR / "dataset_tiled"))
    print(f"tiling {dataset_dir} -> {out_root}  (tile {TILE}, overlap {OVERLAP})")
    for split in ("train", "valid", "test"):
        sd = dataset_dir / split
        if not (sd / "_annotations.coco.json").exists():
            raise SystemExit(f"missing {sd} - run the split step first")
        ni, na, ek, ed = tile_split(sd, out_root / split)
        box_per = na / max(1, ni)
        print(f"  {split:5}: {ni:5d} tiles, {na:6d} boxes ({box_per:.0f}/tile), "
              f"background tiles kept {ek} / dropped {ed}")
    print("Done. Train with:  python scripts/model/train_rfdetr.py "
          f"--dataset-dir {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.dataset_dir, a.out)
