"""Phase 1 - step 2 ("annotator"): convert the base datasets to ONE unified COCO dataset.

Angelina and DSBI have different native annotation formats; we normalize both to the
63-class six-dot COCO schema (see common/braille.py) and merge into a single COCO
dataset at scripts/data/coco/ (images/ + _annotations.coco.json).

Status:
  - DSBI parser: implemented from the real recto.txt grid format (verify with `visualize`).
  - Angelina parser: best-effort LabelMe(.json)/.csv reader. The label ENCODING still needs
    confirming from a real sample -> run `--steps inspect --datasets angelina` and check
    _label_to_class / ANGELINA_LABEL_* below.

Verify visually after building:
    python scripts/dataset/orchestrator.py --steps annotate,visualize --datasets dsbi
    # then open scripts/data/coco/_viz/*.jpg and confirm boxes land on the cells.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import braille, config  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None


# --------------------------------------------------------------------------------------
# COCO assembly
# --------------------------------------------------------------------------------------
class CocoBuilder:
    def __init__(self):
        self.images, self.annotations = [], []
        self._img_id = self._ann_id = 0

    def add_image(self, width, height, file_name, source) -> int:
        self._img_id += 1
        self.images.append({"id": self._img_id, "width": int(width), "height": int(height),
                            "file_name": file_name, "source": source})
        return self._img_id

    def add_box(self, image_id, class_id, xywh) -> None:
        x, y, w, h = (float(v) for v in xywh)
        if w <= 0 or h <= 0:
            return
        self._ann_id += 1
        self.annotations.append({"id": self._ann_id, "image_id": image_id,
                                "category_id": int(class_id), "bbox": [x, y, w, h],
                                "area": w * h, "iscrowd": 0})

    def to_coco(self) -> dict:
        return {"info": {"description": "Unified braille detection dataset (Angelina + DSBI)",
                        "schema": "63-class six-dot encoding"},
                "licenses": [], "images": self.images, "annotations": self.annotations,
                "categories": braille.coco_categories()}


def _require_pillow():
    if Image is None:
        raise SystemExit("Pillow not installed - `pip install -r scripts/requirements.txt`")


def _image_size(path: Path):
    _require_pillow()
    with Image.open(path) as im:
        return im.size  # (w, h)


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS and ".git" not in p.parts:
            yield p


# --------------------------------------------------------------------------------------
# Inspection - reveals the real on-disk format (shows one file per annotation type)
# --------------------------------------------------------------------------------------
def inspect(name: str, max_lines: int = 18) -> None:
    root = Path(config.DATASETS[name]["dir"])
    print(f"\n===== INSPECT: {name}  ({root}) =====")
    if not root.exists():
        print("  NOT DOWNLOADED yet - run the download step first.")
        return
    print("  top-level entries:")
    for p in sorted(root.iterdir()):
        print(f"    [{'d' if p.is_dir() else 'f'}] {p.name}")
    exts = {}
    for p in root.rglob("*"):
        if p.is_file():
            exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
    print("  file extensions:", dict(sorted(exts.items(), key=lambda kv: -kv[1])))

    for ext in (".json", ".csv", ".txt", ".xml"):
        f = next((p for p in sorted(root.rglob(f"*{ext}"))
                  if p.is_file() and ".git" not in p.parts), None)
        if not f:
            continue
        print(f"\n  --- sample {ext}: {f.relative_to(root)} ---")
        if ext == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
                print("   json keys:", list(data)[:12])
                shapes = data.get("shapes")
                if isinstance(shapes, list) and shapes:
                    print("   #shapes:", len(shapes),
                          "imageWidth:", data.get("imageWidth"),
                          "imageHeight:", data.get("imageHeight"))
                    print("   first shape:", json.dumps(shapes[0])[:400])
            except Exception as e:  # noqa: BLE001
                print("   (json parse failed:", e, ")")
        else:
            head = f.read_text(encoding="utf-8", errors="replace").splitlines()[:max_lines]
            print("   " + "\n   ".join(head))
    print("\n  -> Confirm formats, then check parse_angelina_item / parse_dsbi_item.\n")


# --------------------------------------------------------------------------------------
# Angelina parser  (CONFIRMED from real samples)
# --------------------------------------------------------------------------------------
# Each image X.labeled.jpg has a paired X.labeled.csv AND X.labeled.json.
#   - CSV : "left;top;right;bottom;label", coords NORMALIZED [0,1], label = the integer
#           six-dot class id 1..63 (SAME bit convention as ours).
#   - JSON: LabelMe, absolute-pixel coords, but label = the Cyrillic CHARACTER (e.g.
#           "к"), which would need a Russian-braille char table to decode.
# We use the CSV: its integer label IS our class id -> unambiguous. Verified: CSV line 1
# label 5 == JSON shape 1 char "k"(cyrillic) == dots 1,3 == class 5.
def parse_angelina_item(image_path: Path):
    """(w, h, [(class_id, [x,y,w,h]), ...]) for one Angelina image, or None."""
    csv = image_path.with_suffix(".csv")
    if not csv.exists():
        return None
    w, h = _image_size(image_path)
    boxes, bad = [], []
    for line in csv.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().replace(";", ",").split(",")
        if len(parts) < 5:
            continue
        try:
            l, t, r, b = (float(v) for v in parts[:4])
            cid = int(float(parts[4]))
        except ValueError:
            continue
        if not 1 <= cid <= 63:
            bad.append(parts[4])
            continue
        x1, y1, x2, y2 = l * w, t * h, r * w, b * h
        boxes.append((cid, [x1, y1, x2 - x1, y2 - y1]))
    if bad:
        print(f"    [angelina] {image_path.name}: skipped {len(bad)} labels outside 1..63 "
              f"e.g. {sorted(set(bad))[:8]}")
    return w, h, boxes


# --------------------------------------------------------------------------------------
# DSBI parser  (implemented from the real recto.txt grid format)
# --------------------------------------------------------------------------------------
# recto.txt layout:
#   line 0: skew angle (float)
#   line 1: x-centers of each cell's two dot-columns, left->right  (2 per grid column)
#   line 2: y-centers of each cell's three dot-rows, top->bottom   (3 per grid row)
#   line 3..: "row col d1 d2 d3 d4 d5 d6"  (1-based row/col; six 0/1 dot flags)
# A cell box spans its two dot-columns x its three dot-rows, padded by ~0.6 * dot spacing.
def parse_dsbi_item(image_path: Path):
    recto = image_path.parent / (image_path.stem + "+recto.txt")
    if not recto.exists():
        return None  # not a DSBI page image (e.g. a figure) - skip
    w, h = _image_size(image_path)
    lines = [ln.strip() for ln in recto.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    if len(lines) < 4:
        return None
    try:
        xs = [int(round(float(v))) for v in lines[1].split()]
        ys = [int(round(float(v))) for v in lines[2].split()]
    except ValueError:
        return None
    pad_x = max(4, int((xs[1] - xs[0]) * 0.6)) if len(xs) >= 2 else 8
    pad_y = max(4, int((ys[1] - ys[0]) * 0.6)) if len(ys) >= 2 else 8

    boxes = []
    for ln in lines[3:]:
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            row, col = int(parts[0]), int(parts[1])
            flags = [int(v) for v in parts[2:8]]
        except ValueError:
            continue
        dots = [i + 1 for i, f in enumerate(flags) if f]
        if not dots:
            continue  # empty cell
        xi, yi = 2 * (col - 1), 3 * (row - 1)
        if xi < 0 or yi < 0 or xi + 1 >= len(xs) or yi + 2 >= len(ys):
            continue
        left_x, right_x = xs[xi], xs[xi + 1]
        top_y, bot_y = ys[yi], ys[yi + 2]
        x1, y1 = max(0, left_x - pad_x), max(0, top_y - pad_y)
        x2, y2 = min(w, right_x + pad_x), min(h, bot_y + pad_y)
        try:
            cid = braille.dots_to_class(dots)
        except ValueError:
            continue
        boxes.append((cid, [x1, y1, x2 - x1, y2 - y1]))
    return w, h, boxes


PARSERS = {"angelina": parse_angelina_item, "dsbi": parse_dsbi_item}


# --------------------------------------------------------------------------------------
# Build unified COCO
# --------------------------------------------------------------------------------------
def build(datasets=None, out_dir: Path = None, copy_images: bool = True) -> Path:
    datasets = datasets or list(config.DATASETS)
    out_dir = Path(out_dir or config.COCO_DIR)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    builder = CocoBuilder()
    for name in datasets:
        parser = PARSERS[name]
        root = Path(config.DATASETS[name]["dir"])
        if not root.exists():
            print(f"[{name}] not downloaded - skipping")
            continue
        n_img = n_box = n_skip = 0
        for img_path in _iter_images(root):
            try:
                parsed = parser(img_path)
            except NotImplementedError as e:
                print(f"[{name}] parser not ready: {e}")
                break
            if parsed is None:
                continue
            w, h, boxes = parsed
            if not boxes:
                n_skip += 1
                continue
            file_name = f"{name}__{img_path.name}"
            if copy_images:
                shutil.copy2(img_path, img_out / file_name)
            img_id = builder.add_image(w, h, file_name, source=name)
            for cid, xywh in boxes:
                builder.add_box(img_id, cid, xywh)
                n_box += 1
            n_img += 1
        print(f"[{name}] {n_img} images, {n_box} boxes, {n_skip} images without boxes")

    ann_path = out_dir / "_annotations.coco.json"
    ann_path.write_text(json.dumps(builder.to_coco()), encoding="utf-8")
    print(f"\nWrote {ann_path}")
    print(f"Totals: {len(builder.images)} images, {len(builder.annotations)} boxes")
    if not builder.images:
        print("WARNING: 0 images - parsers likely don't match the real layout; run inspect.")
    return out_dir


# --------------------------------------------------------------------------------------
# Visualize - draw the produced COCO boxes back onto images to verify correctness
# --------------------------------------------------------------------------------------
def visualize(coco_dir: Path = None, n: int = 8) -> None:
    _require_pillow()
    coco_dir = Path(coco_dir or config.COCO_DIR)
    ann_path = coco_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print("No COCO file - run the annotate step first.")
        return
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    if not imgs:
        print("COCO has 0 images.")
        return
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    label = {c["id"]: (c.get("letter") or c["name"]) for c in coco["categories"]}
    out = coco_dir / "_viz"
    out.mkdir(exist_ok=True)
    ids = list(imgs)
    step = max(1, len(ids) // max(1, n))
    for iid in ids[::step][:n]:
        meta = imgs[iid]
        p = coco_dir / "images" / meta["file_name"]
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        draw = ImageDraw.Draw(img)
        for a in by_img.get(iid, []):
            x, y, bw, bh = a["bbox"]
            draw.rectangle([x, y, x + bw, y + bh], outline=(255, 0, 0), width=2)
            draw.text((x + 1, max(0, y - 10)), str(label.get(a["category_id"], "?")),
                      fill=(255, 255, 0))
        img.save(out / meta["file_name"])
    print(f"Wrote up to {n} visualized images to {out}\n"
          f"-> open them and confirm boxes sit on the braille cells.")


def main(datasets=None) -> None:
    build(datasets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of: " + " ".join(config.DATASETS))
    ap.add_argument("--inspect", action="store_true", help="print raw structure and exit")
    ap.add_argument("--visualize", action="store_true", help="draw COCO boxes on images and exit")
    args = ap.parse_args()
    if args.inspect:
        for d in (args.datasets or list(config.DATASETS)):
            inspect(d)
    elif args.visualize:
        visualize()
    else:
        main(args.datasets)
