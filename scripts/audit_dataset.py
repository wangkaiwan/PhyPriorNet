"""Audit the downloaded photon dataset and print a fact sheet:
patient count, per-site counts, control-point counts, image geometry,
dose magnitude range, and presence/absence of OAR contours.

Usage: conda run -n doserad python scripts/audit_dataset.py [training_root]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from doserad.data.index import build_index, patient_site
from doserad.io.mha import load_mha

DEFAULT_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


def main(root: str = DEFAULT_ROOT) -> None:
    root = Path(root)
    patients = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"patients: {len(patients)}")
    print("by site:", dict(Counter(patient_site(p) for p in patients)))

    df = build_index(root)
    print(f"index rows (patient,beam,cp): {len(df)}")
    print("control points per patient:",
          df.groupby('patient_id').size().describe().to_dict())

    # geometry + dose magnitude on the first patient
    first = patients[0]
    ct = load_mha(root / first / "image" / "ct.mha")
    print(f"[{first}] ct shape={ct.array.shape} spacing={ct.spacing}")
    dose_path = df[df.patient_id == first].iloc[0]["dose_path"]
    dose = load_mha(dose_path)
    print(f"[{first}] dose max={dose.array.max():.3e} mean={dose.array.mean():.3e}")

    # OAR contours? look for any 'struct'/'rtss'/'mask' dirs or files
    # Note: use any(...) on the generator to actually iterate matches,
    # not on a generator object itself (which is always truthy).
    has_struct = any(
        any((root / p).glob(pat))
        for pat in ("**/*struct*", "**/*mask*", "**/*rtss*")
        for p in patients[:5]
    )
    print(f"OAR contours present (heuristic): {bool(has_struct)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
