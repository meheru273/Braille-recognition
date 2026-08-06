"""Phase 3 - split the unified COCO dataset into the RF-DETR train/valid/test layout.

LEAKAGE RULE (the #1 validity issue for the paper): the split unit is the GROUP
(source book/document folder), never the image and never the cell. Pages of one book
share paper, embossing depth and lighting - putting them on both sides of the split
inflates test mAP. Uses sklearn GroupShuffleSplit with a fixed seed; the resulting
assignment is saved to splits.json so the exact split ships with the dataset.

Output layout (exactly what rfdetr's .train(dataset_dir=...) expects):
    dataset/
      train/ _annotations.coco.json + images
      valid/ _annotations.coco.json + images
      test/  _annotations.coco.json + images

    python scripts/dataset/split.py                     # 70/15/15 by group, seed 0
    python scripts/dataset/split.py --test-size 0.2 --valid-size 0.1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402

DATASET_DIR = config.DATA_DIR / "dataset"


def _group_of(im: dict, negative_ids=frozenset()) -> str:
    # 'group' exists in COCO built after the group-metadata change; fall back to a
    # name-derived group so older _annotations.coco.json files still split safely.
    g = im.get("group") or im["file_name"].split("__")[0]
    # Hard negatives (e.g. angelina/not_braille/misc) are UNRELATED photos, not pages
    # of one document - there is no shared-source leakage between them. Treating the
    # folder as one group would dump all negatives into a single split (seed 0 put all
    # 44 in test), so each negative image is its own group and they spread naturally.
    if im["id"] in negative_ids:
        return f"{g}/{im['file_name']}"
    return g


# Default seed chosen by a constrained search over seeds 0-39 (see git history):
# it is the seed whose grouped split best satisfies (a) both sources (angelina+dsbi)
# present in valid AND test, (b) negatives present in every split (31/6/7), and
# (c) test share of boxes closest to 15% (exactly 0.15). Seed 0 - the naive choice -
# put ALL 44 negatives in test and zero DSBI outside train.
DEFAULT_SEED = 8


def split(coco_dir: Path = None, out_dir: Path = None, test_size: float = 0.15,
          valid_size: float = 0.15, seed: int = DEFAULT_SEED) -> None:
    from sklearn.model_selection import GroupShuffleSplit

    coco_dir = Path(coco_dir or config.COCO_DIR)
    out_dir = Path(out_dir or DATASET_DIR)
    coco = json.loads((coco_dir / "_annotations.coco.json").read_text(encoding="utf-8"))

    images = coco["images"]
    with_boxes = {a["image_id"] for a in coco["annotations"]}
    negative_ids = frozenset(im["id"] for im in images if im["id"] not in with_boxes)
    groups = [_group_of(im, negative_ids) for im in images]
    idx = list(range(len(images)))

    # 1st cut: test off the rest; 2nd cut: valid off the remainder
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest, test = next(gss1.split(idx, groups=groups))
    rest_groups = [groups[i] for i in rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size / (1 - test_size),
                             random_state=seed)
    tr, va = next(gss2.split(rest, groups=rest_groups))
    assign = {}
    for i in test:
        assign[images[i]["id"]] = "test"
    for j in va:
        assign[images[rest[j]]["id"]] = "valid"
    for j in tr:
        assign[images[rest[j]]["id"]] = "train"

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    stats = {}
    for name in ("train", "valid", "test"):
        sub_dir = out_dir / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_imgs = [im for im in images if assign[im["id"]] == name]
        sub_anns = []
        for im in sub_imgs:
            for a in anns_by_img[im["id"]]:
                a2 = dict(a)
                a2.pop("score", None)          # pre-label metadata, not GT
                sub_anns.append(a2)
            src = coco_dir / "images" / im["file_name"]
            if src.exists():
                shutil.copy2(src, sub_dir / im["file_name"])
        sub = {"info": coco.get("info", {}), "licenses": coco.get("licenses", []),
               "images": sub_imgs, "annotations": sub_anns,
               "categories": coco["categories"]}
        (sub_dir / "_annotations.coco.json").write_text(json.dumps(sub), encoding="utf-8")
        stats[name] = (len(sub_imgs), len(sub_anns),
                       sorted({_group_of(im, negative_ids) for im in sub_imgs}))

    # ship the exact split with the dataset for reproducibility
    (out_dir / "splits.json").write_text(json.dumps(
        {"seed": seed, "test_size": test_size, "valid_size": valid_size,
         "assignment": {im["file_name"]: assign[im["id"]] for im in images}},
        indent=1), encoding="utf-8")

    print(f"\n=== Split by GROUP (seed={seed}) -> {out_dir} ===")
    for name, (ni, na, grps) in stats.items():
        sub_imgs = [im for im in images if assign[im["id"]] == name]
        n_neg = sum(1 for im in sub_imgs if im["id"] in negative_ids)
        srcs = sorted({im.get("source", "?") for im in sub_imgs})
        print(f"  {name:5}: {ni:3d} images ({n_neg} neg), {na:6d} boxes, "
              f"{len(grps):2d} groups, sources={srcs}")
        for g in sorted({gr for gr in grps if "not_braille" not in gr}):
            print(f"         - {g}")
        n_neg_groups = sum(1 for gr in grps if "not_braille" in gr)
        if n_neg_groups:
            print(f"         - ({n_neg_groups} individual not_braille negatives)")
    # sanity: no group in two splits
    seen = {}
    for name, (_, _, grps) in stats.items():
        for g in grps:
            assert g not in seen, f"LEAK: group {g} in {seen[g]} and {name}"
            seen[g] = name
    print("  OK: no group appears in more than one split.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=None,
                    help=f"group-shuffle seed (default {DEFAULT_SEED}, see comment)")
    a = ap.parse_args()
    split(a.coco_dir, a.out_dir, a.test_size, a.valid_size,
          DEFAULT_SEED if a.seed is None else a.seed)
