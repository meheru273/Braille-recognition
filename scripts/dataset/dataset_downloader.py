"""Phase 1 - step 1: download the base braille datasets.

Clones AngelinaDataset and DSBI from GitHub into scripts/data/raw/. Run on any
machine with internet access (including your remote GPU box). Idempotent: if a
dataset is already cloned it is fast-forward pulled instead of re-cloned.

    python scripts/dataset/dataset_downloader.py                 # both
    python scripts/dataset/dataset_downloader.py --datasets angelina
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402


def _run(cmd) -> None:
    print("   $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def clone_or_update(name: str) -> Path:
    spec = config.DATASETS[name]
    dest = Path(spec["dir"])
    if (dest / ".git").exists():
        print(f"[{name}] present at {dest} -> pulling latest")
        try:
            _run(["git", "-C", str(dest), "pull", "--ff-only"])
        except subprocess.CalledProcessError:
            print(f"[{name}] pull failed (local changes?) - keeping existing checkout")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] cloning {spec['repo']}")
        _run(["git", "clone", "--depth", "1", spec["repo"], str(dest)])
    return dest


def main(datasets=None) -> None:
    datasets = datasets or list(config.DATASETS)
    unknown = [d for d in datasets if d not in config.DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}. Known: {list(config.DATASETS)}")
    for name in datasets:
        path = clone_or_update(name)
        n_files = sum(1 for p in path.rglob("*") if p.is_file())
        print(f"[{name}] done - {n_files} files under {path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of: " + " ".join(config.DATASETS))
    main(ap.parse_args().datasets)
