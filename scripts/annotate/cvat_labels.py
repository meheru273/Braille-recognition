"""Generate the CVAT "raw" labels JSON for the 63-class six-dot schema.

CVAT can only import our COCO pre-labels if the project's label NAMES match the
COCO category names (dot-strings '1'..'123456'). Creating 63 labels by hand in the
UI is error-prone, so paste the generated JSON into CVAT instead:
Project -> Constructor -> Raw -> paste -> Done.

    python scripts/annotate/cvat_labels.py          # writes cvat_labels.json + prints path
"""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import braille, config  # noqa: E402


def make_labels() -> list:
    labels = []
    for i, cat in enumerate(braille.coco_categories()):
        # deterministic distinct colors around the hue wheel
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95)
        labels.append({
            "name": cat["name"],                       # dot-string, e.g. "145"
            "color": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
            "type": "rectangle",
            "attributes": [],
        })
    return labels


if __name__ == "__main__":
    out = Path(config.CONTRIB_DIR) / "cvat_labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(make_labels(), indent=1), encoding="utf-8")
    print(f"wrote {len(make_labels())} labels -> {out}")
    print("CVAT: Project -> Constructor -> Raw -> paste the file contents -> Done")
