"""Build the patient-level split files.

    python scripts/build_splits.py            # 5-fold, site-stratified -> $WORKDIR/splits_final.json
    python scripts/build_splits.py --all75    # every patient in train  -> $WORKDIR/splits_all75.json

The released models are trained with --all75 (the challenge leaderboard was our held-out signal);
the 5-fold file reproduces the cross-validation protocol reported in the papers.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import os

import json
from pathlib import Path

from doserad.data.index import patient_site
from doserad.data.splits import make_kfold_splits

DEFAULT_ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training")
OUT = Path((os.environ.get("WORKDIR", "./workdir") + "/splits_final.json"))


ALL75 = Path((os.environ.get("WORKDIR", "./workdir") + "/splits_all75.json"))


def main(root: str = DEFAULT_ROOT, all75: bool = False) -> None:
    patients = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    if not patients:
        raise SystemExit(f"no patients under {root}; set DATA_ROOT to the challenge data")
    if all75:
        ALL75.parent.mkdir(parents=True, exist_ok=True)
        ALL75.write_text(json.dumps({"train": patients, "val": []}, indent=2))
        print(f"wrote {ALL75} with all {len(patients)} patients in train")
        return
    sites = {p: patient_site(p) for p in patients}
    folds = make_kfold_splits(patients, sites, k=5, seed=42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(folds, indent=2))
    print(f"wrote {OUT} with {len(patients)} patients across 5 folds")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="photon training root (patient dirs)")
    ap.add_argument("--all75", action="store_true", help="write splits_all75.json instead of k-fold")
    a = ap.parse_args()
    main(a.root, a.all75)
