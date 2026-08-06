"""Shared configuration for the braille research pipeline.

Paths are computed relative to this file so the whole `scripts/` folder can be
copied to a remote GPU box (via AnyDesk) and still work. Data lives *inside*
`scripts/data/` by default so one folder copy is self-contained; override with
the BRAILLE_DATA_DIR environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent          # .../scripts
REPO_ROOT = SCRIPTS_DIR.parent                                # repo root

DATA_DIR = Path(os.environ.get("BRAILLE_DATA_DIR", SCRIPTS_DIR / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"          # cloned source datasets
COCO_DIR = DATA_DIR / "coco"        # unified COCO output (Phase 1)

# Phase 2 - our own contributed photos
CONTRIB_DIR = DATA_DIR / "contrib"
CONTRIB_ENHANCED = CONTRIB_DIR / "enhanced"
CONTRIB_CROPPED = CONTRIB_DIR / "cropped"
CONTRIB_PRELABEL = CONTRIB_DIR / "prelabel"
CONTRIB_PREVIEW = CONTRIB_DIR / "preview"

# Our DSLR photos are ~24MP with braille cells ~85px wide, while the base datasets
# (Angelina) have cells ~28px. Detectors are scale-sensitive, so cropped pages are
# resized so a cell lands near this width - matching the training distribution.
TARGET_CELL_PX = 28
APPROX_SOURCE_CELL_PX = 85          # measured on the DSC*.jpg set; re-measure if the rig changes

# Base datasets (Phase 1). See RESEARCH_PLAN.md.
#  - angelina: page photos, per-character boxes, 1-63 six-dot labels (Ovodov, ICCVW 2021)
#  - dsbi:     114 double-sided images, per-cell recto/verso dot-state labels
DATASETS = {
    "angelina": {
        "repo": "https://github.com/IlyaOvodov/AngelinaDataset.git",
        "dir": RAW_DIR / "angelina",
    },
    "dsbi": {
        "repo": "https://github.com/yeluo1994/DSBI.git",
        "dir": RAW_DIR / "dsbi",
    },
}

# 63-class six-dot encoding (decision D2). Class ids = dot bitmask 1..63.
NUM_CLASSES = 63

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
