"""Build the data manifest for the classify-then-regress sCT pipeline (lever 4).

Source = SynthRAD2025 Task1 (paired MR/CT/mask), regions AB + TH only (HN excluded: all high-field,
no 0.35T, different anatomy). Patient ID = <task><region><center><id>, so the 4th char is the center;
center B = 0.35T ViewRay MRIdian = the DoseRAD2026 challenge cohort (our deployment field strength).

Two training pools (the 16 challenge val are held out from BOTH; matches the v4 leak-free policy):
  - classifier pool : ALL centers AB+TH  (classification is field-robust -> use all data for better
                      bone/lung localisation)
  - refiner pool    : Center-B (0.35T) only  (fine HU regression is field-sensitive -> field-matched)

CAVEAT (flagged, not resolved here): the 107 Center-B cases not in the challenge train dir *might*
include the challenge's hidden test. We exclude only the known 16 val (v4 policy). Revisit against the
challenge rules before final submission.

    conda run -n doserad python scripts/build_sct_classify_manifest.py
"""
from __future__ import annotations
import json, os
from pathlib import Path

SR = Path("/data/kwang/synthrad2025_data/synthRAD2025_Task1_Train/Task1")
CHAL = Path("/data/kwang/DoseRad2026_raw/photon/training")
SPLITS = Path("/home/kaiwang/doserad2026_workdir/splits_final.json")
OUT = Path("/home/kaiwang/doserad2026_workdir/sct_manifest.json")
REGIONS = ("AB", "TH")                       # HN excluded


def field_of(center: str) -> str:
    return "0.35T" if center == "B" else "high"   # B=0.35T MR-Linac; A/C/D = 1.5/3T


def main():
    val = set(json.load(open(SPLITS))["fold_0"]["val"])
    cases = []
    for region in REGIONS:
        rdir = SR / region
        if not rdir.is_dir():
            continue
        for d in sorted(rdir.iterdir()):
            pid = d.name
            if not d.is_dir() or len(pid) < 4:
                continue
            mr, ct, mask = d / "mr.mha", d / "ct.mha", d / "mask.mha"
            if not (mr.exists() and ct.exists()):
                continue
            center = pid[3]
            cases.append({
                "pid": pid, "region": region, "center": center, "field": field_of(center),
                "mr": str(mr), "ct": str(ct), "mask": str(mask) if mask.exists() else None,
                "in_challenge_dir": (CHAL / pid).is_dir(),
                "is_val": pid in val,
            })

    train = [c for c in cases if not c["is_val"]]
    valc = [c for c in cases if c["is_val"]]
    classifier_pool = [c["pid"] for c in train]                       # all centers
    refiner_pool = [c["pid"] for c in train if c["field"] == "0.35T"]  # Center-B only
    val_pids = [c["pid"] for c in valc]

    manifest = {
        "regions": list(REGIONS),
        "cases": {c["pid"]: c for c in cases},
        "classifier_train": classifier_pool,
        "refiner_train": refiner_pool,
        "val": val_pids,
        "classes": {  # CT-HU band -> class id; bulk density / representative HU for the coarse prior
            "names": ["air", "lung", "soft", "bone"],
            "hu_bands": [[-100000, -700], [-700, -300], [-300, 200], [200, 100000]],
            "bulk_rho": [0.0012, 0.30, 1.00, 1.50],
            "rep_hu": [-1000.0, -600.0, 30.0, 700.0],
        },
    }
    OUT.write_text(json.dumps(manifest, indent=2))

    # --- summary ---
    def tally(pids):
        d = {}
        for p in pids:
            c = manifest["cases"][p]
            k = f'{c["region"]}-{c["center"]}({c["field"]})'
            d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items()))

    print(f"total AB+TH cases: {len(cases)}  (val held out: {len(val_pids)})")
    print(f"classifier_train (all centers): {len(classifier_pool)}  -> {tally(classifier_pool)}")
    print(f"refiner_train   (0.35T only)  : {len(refiner_pool)}  -> {tally(refiner_pool)}")
    print(f"val: {len(val_pids)} -> {tally(val_pids)}")
    miss = [c['pid'] for c in cases if c['mask'] is None]
    print(f"cases without mask.mha: {len(miss)}" + (f"  e.g. {miss[:3]}" if miss else ""))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
