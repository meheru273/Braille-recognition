"""Phase 1 orchestrator: run the dataset-build steps one by one, in order.

Designed to be EXTENDED later (Phase 2/3: enhance, crop, prelabel, split, ...). Each
step is a named callable in the STEPS registry; add a function and register it.

    python scripts/dataset/orchestrator.py                      # default pipeline
    python scripts/dataset/orchestrator.py --list               # show steps
    python scripts/dataset/orchestrator.py --steps download,inspect
    python scripts/dataset/orchestrator.py --datasets angelina
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402
from dataset import annotator, dataset_downloader  # noqa: E402


def step_download(datasets):
    dataset_downloader.main(datasets)


def step_inspect(datasets):
    for name in (datasets or list(config.DATASETS)):
        annotator.inspect(name)


def step_annotate(datasets):
    annotator.main(datasets)


def step_visualize(datasets):
    annotator.visualize()


def step_verify(datasets):
    annotator.verify_angelina()


def step_stats(datasets):
    annotator.stats()


# ordered registry - extend here for future phases
STEPS = {
    "download": step_download,
    "inspect": step_inspect,
    "verify": step_verify,
    "annotate": step_annotate,
    "visualize": step_visualize,
    "stats": step_stats,
}
DEFAULT_PIPELINE = ["download", "annotate"]


def run(steps, datasets=None):
    for name in steps:
        if name not in STEPS:
            raise SystemExit(f"Unknown step '{name}'. Available: {list(STEPS)}")
    for name in steps:
        print(f"\n########## STEP: {name} ##########")
        STEPS[name](datasets)
    print("\nPipeline complete:", " -> ".join(steps))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(DEFAULT_PIPELINE),
                    help="comma-separated, in order. Default: " + ",".join(DEFAULT_PIPELINE))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of: " + " ".join(config.DATASETS))
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    args = ap.parse_args()
    if args.list:
        print("Available steps (in registry order):")
        for s in STEPS:
            print("  -", s)
        print("Default pipeline:", " -> ".join(DEFAULT_PIPELINE))
        raise SystemExit(0)
    run([s.strip() for s in args.steps.split(",") if s.strip()], args.datasets)
