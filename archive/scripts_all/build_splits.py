"""Build splits_final.json (5-fold, patient-level, site-stratified)."""
from __future__ import annotations

import json
from pathlib import Path

from doserad.data.index import patient_site
from doserad.data.splits import make_kfold_splits

DEFAULT_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
OUT = Path("/home/kaiwang/doserad2026_workdir/splits_final.json")


def main(root: str = DEFAULT_ROOT) -> None:
    patients = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    sites = {p: patient_site(p) for p in patients}
    folds = make_kfold_splits(patients, sites, k=5, seed=42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(folds, indent=2))
    print(f"wrote {OUT} with {len(patients)} patients across 5 folds")


if __name__ == "__main__":
    main()
