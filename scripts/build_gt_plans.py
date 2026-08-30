"""Cache the TRUE full-grid ground-truth plan dose, one array per patient.

Every internal number we have ever quoted built the GT plan by summing the CACHED per-CP crops --
the same aperture-bbox+margin windows our prediction uses -- so prediction and reference were
truncated identically and the metric could not see what the crop dropped. Scored against the raw
full-grid GT instead, one patient went from 96.18 to 86.38 plan gamma.

Per-CP metrics are unaffected (0.00% of a CP's >=10%-of-max voxels fall outside its crop); only the
PLAN-level metrics are, because the plan sums 540 CPs and each contributes a low-dose tail outside
its own crop. So the fix is exactly this: a true plan-level reference.

Reading 540 raw .mha files per patient is ~32 GB of I/O, far too slow to repeat per evaluation, so
accumulate once and cache the plan (~60 MB/patient, ~4.5 GB for the cohort).

  python scripts/build_gt_plans.py --particle photon [--out DIR] [--patients PID,PID]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW = "/data/kwang/DoseRad2026_raw"
DEFAULT_OUT = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir")) / "cache" / "gt_plans"


def build(particle: str, out_dir: Path, patients=None, force=False):
    root = Path(RAW) / particle / "training"
    pids = patients or sorted(p.name for p in root.iterdir() if (p / "dose").is_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(pids, 1):
        dst = out_dir / f"{pid}.npy"
        if dst.exists() and not force:
            print(f"[{i}/{len(pids)}] {pid}: exists, skip", flush=True)
            continue
        files = sorted(glob.glob(str(root / pid / "dose" / "*.mha")))
        if not files:
            print(f"[{i}/{len(pids)}] {pid}: no dose files, skip", flush=True)
            continue
        acc = None
        for f in files:
            a = sitk.GetArrayFromImage(sitk.ReadImage(f))
            acc = a.astype(np.float32) if acc is None else acc + a
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, acc)
        tmp.replace(dst)
        print(f"[{i}/{len(pids)}] {pid}: {len(files)} units -> {acc.shape} "
              f"max {acc.max():.4e} ({dst.stat().st_size/2**20:.0f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle", default="photon", choices=["photon", "proton"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--patients", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else DEFAULT_OUT / a.particle
    build(a.particle, out, a.patients.split(",") if a.patients else None, a.force)
