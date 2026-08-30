"""Precompute photon physics channels (on CT) for cached training reads.

Writes float16 .npz to $DOSERAD_WORK/cache/physics/photon/<pid>/<B>_<CP>.npz
with key 'channels' shape (5, z, y, x).

Usage: conda run -n doserad python scripts/precompute_photon_channels.py [pid ...]
       (no args = all patients in the index)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.data.index import build_index
from doserad.io.mha import load_mha
from doserad.physics.channels import photon_channels
from doserad.physics.machine import load_photon_machine

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")


def process_patient(pid: str) -> int:
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    out_dir = WORK / "cache" / "physics" / "photon" / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for beam in plan.beams:
        for cp in beam.control_points:
            out = out_dir / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if out.exists():
                continue
            chans = photon_channels(
                image=ct, machine=MACHINE, iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm))
            np.savez_compressed(out, channels=chans.astype(np.float16))
            n += 1
    return n


def main(pids):
    if not pids:
        pids = sorted(build_index(ROOT).patient_id.unique())
    for pid in pids:
        n = process_patient(pid)
        print(f"{pid}: wrote {n} channel files")


if __name__ == "__main__":
    main(sys.argv[1:])
