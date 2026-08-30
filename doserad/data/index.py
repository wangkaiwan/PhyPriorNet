"""Enumerate the photon training tree into a flat (patient, beam, cp) index."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from doserad.beam.parse import load_photon_plan

_SITE_RE = re.compile(r"^\d+([A-Za-z]+?)\d+$")


def patient_site(patient_id: str) -> str:
    """Extract the alpha site code from an ID like '1ABB006' -> 'ABB'."""
    m = _SITE_RE.match(patient_id)
    return m.group(1) if m else "UNK"


def build_index(training_root: str | Path) -> pd.DataFrame:
    training_root = Path(training_root)
    rows = []
    for pdir in sorted(p for p in training_root.iterdir() if p.is_dir()):
        pid = pdir.name
        json_path = pdir / f"{pid}.json"
        if not json_path.exists():
            continue
        plan = load_photon_plan(json_path)
        ct = pdir / "image" / "ct.mha"
        mr = pdir / "image" / "mr.mha"
        for beam in plan.beams:
            for cp in beam.control_points:
                dose = pdir / "dose" / f"Dose_B{beam.beam_idx}_CP{cp.cp_idx:03d}.mha"
                rows.append({
                    "patient_id": pid,
                    "site": patient_site(pid),
                    "beam_idx": beam.beam_idx,
                    "cp_idx": cp.cp_idx,
                    "gantry_angle": cp.gantry_angle,
                    "ct_path": str(ct),
                    "mr_path": str(mr),
                    "dose_path": str(dose),
                    "json_path": str(json_path),
                })
    return pd.DataFrame(rows)
