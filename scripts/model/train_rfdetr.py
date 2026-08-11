"""Phase 4 - fine-tune RF-DETR on the braille dataset (63-class six-dot schema).

Needs: pip install rfdetr  (Python >=3.10, CUDA GPU; use rfdetr[loggers] for
TensorBoard/W&B - note there is NO rfdetr[metrics] extra).

Dataset layout (produced by scripts/dataset/split.py):
    dataset/{train,valid,test}/ each with images + _annotations.coco.json

IMPORTANT: --dataset-dir defaults to the UNTILED dataset/. If you ran
scripts/dataset/tile.py, you must explicitly pass
--dataset-dir scripts/data/dataset_tiled to actually train on tiles - nothing
selects it automatically. (This bit us once: a retrain silently reused the
untiled set because the flag was omitted; per-epoch wall-clock time was the
tell - tiles are ~8x more images and should take ~8x longer per epoch.)

Defaults follow the plan (RESEARCH_PLAN.md): Medium@576, effective batch 16
(batch 4 x grad-accum 4 for a 16GB GPU), lr 1e-4, ~50 epochs, report from
checkpoint_best_ema.pth. Stay within Nano-Large (Apache-2.0); never XL/2XL.

    python scripts/model/train_rfdetr.py --dataset-dir scripts/data/dataset
    python scripts/model/train_rfdetr.py --model nano --epochs 30      # quick first run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import config  # noqa: E402

MODELS = {}  # name -> class, resolved lazily so --help works without rfdetr installed


def _load_models():
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    MODELS.update({"nano": RFDETRNano, "small": RFDETRSmall,
                   "medium": RFDETRMedium, "large": RFDETRLarge})


_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee:
    """Mirror a stream into a log file (ANSI-stripped) so the console metric tables
    survive the terminal scrollback. rfdetr prints its per-epoch mAP tables via rich
    to stdout only - without this, a closed terminal means the history is gone."""

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, text):
        self._s.write(text)
        self._f.write(_ANSI.sub("", text))
        self._f.flush()
        return len(text)

    def flush(self):
        self._s.flush()
        self._f.flush()

    def isatty(self):
        return self._s.isatty()

    def __getattr__(self, name):
        return getattr(self._s, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(config.DATA_DIR / "dataset"))
    ap.add_argument("--model", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch-size x grad-accum (target 16)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None,
                    help="must be divisible by 56; default = model's native")
    ap.add_argument("--output", default=str(config.DATA_DIR / "runs" / "rfdetr"))
    ap.add_argument("--no-tensorboard", action="store_true",
                    help="TensorBoard is ON by default (needs rfdetr[train,loggers])")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    out = Path(a.output) / a.model
    out.mkdir(parents=True, exist_ok=True)

    # Tee is armed FIRST - before the dataset check, before model load - so that
    # WHICH dataset/config was used survives a crash. This is what let us catch a
    # retrain silently reusing the untiled dataset: the resolved --dataset-dir and
    # image/box counts below are now always in the log file, not just the console.
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    console_log = out / f"train_console_{stamp}.log"
    fh = open(console_log, "a", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"console log -> {console_log}")

    # Archive a PRIOR run's metrics.csv / tfevents before this run starts, since
    # Lightning (CSVLogger/TensorBoardLogger with name="",version="") overwrites/
    # deletes them in-place otherwise - that silently destroyed the first medium
    # run's logs when a second run reused the same --output.
    prior = [p for p in out.glob("metrics.csv")] + list(out.glob("events.out.tfevents.*"))
    if prior:
        archive = out / f"archived_{stamp}"
        archive.mkdir(exist_ok=True)
        for p in prior:
            p.rename(archive / p.name)
        print(f"archived {len(prior)} log file(s) from a previous run -> {archive}")

    print(f"\n>>> --dataset-dir = {a.dataset_dir}")
    print(">>> (defaults to the UNTILED dataset/ - pass dataset_tiled explicitly if that's intended)\n")

    import json
    ds = Path(a.dataset_dir)
    print("dataset sanity check:")
    for split_name in ("train", "valid", "test"):
        ann = ds / split_name / "_annotations.coco.json"
        if not ann.exists():
            raise SystemExit(f"missing {ann} - run scripts/dataset/split.py first")
        d = json.loads(ann.read_text(encoding="utf-8"))
        n_img, n_ann = len(d["images"]), len(d["annotations"])
        srcs = sorted({im.get("source", "?") for im in d["images"]})
        print(f"  {split_name:5}: {n_img:4d} images, {n_ann:6d} boxes, sources={srcs}")
        # Guard against training on a stale/degenerate split (e.g. the old seed-0 run
        # wrote valid/test containing ONLY negative images before its assert fired).
        if n_ann == 0:
            raise SystemExit(
                f"{split_name} has 0 annotations - this is a stale/broken split. "
                "Re-run: python scripts/dataset/orchestrator.py --steps annotate,split")
    if a.resolution and a.resolution % 56:
        raise SystemExit(f"--resolution must be divisible by 56 (got {a.resolution})")

    _load_models()
    model = MODELS[a.model]()          # COCO-pretrained weights auto-download

    kwargs = dict(dataset_dir=str(ds), epochs=a.epochs, batch_size=a.batch_size,
                  grad_accum_steps=a.grad_accum, lr=a.lr, output_dir=str(out))
    if a.resolution:
        kwargs["resolution"] = a.resolution
    if not a.no_tensorboard:
        kwargs["tensorboard"] = True
    if a.wandb:
        kwargs["wandb"] = True

    print(f"RF-DETR {a.model} | effective batch {a.batch_size * a.grad_accum} | "
          f"epochs {a.epochs} | out {out}")
    print(f"tensorboard -> {'ON (tensorboard --logdir ' + str(out) + ')' if not a.no_tensorboard else 'off'}")
    model.train(**kwargs)
    print(f"\nDone. Report metrics from {out}/checkpoint_best_ema.pth")
    print("Next: prelabel our photos with it ->")
    print(f"  python scripts/annotate/prelabel.py --backend rfdetr "
          f"--weights {out}/checkpoint_best_ema.pth --auto-orient")


if __name__ == "__main__":
    main()
