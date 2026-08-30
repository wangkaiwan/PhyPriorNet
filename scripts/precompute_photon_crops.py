"""Precompute beam-bbox-cropped channels + dose per CP.

Out: $DOSERAD_WORK/cache/crops/photon/<pid>/<B>_<CP>.npz
  channels (5,d,h,w) fp16, dose (d,h,w) fp16, bbox int[6], dose_max float
Usage: conda run -n doserad python scripts/precompute_photon_crops.py [pid ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.data.crop import compute_bbox, crop_to_bbox
from doserad.data.index import build_index
from doserad.io.mha import load_mha
from doserad.physics.channels import photon_channels
from doserad.physics.machine import load_photon_machine

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")
MARGIN = 8
# DOSERAD_FORCE=1 overwrites existing crops (needed after a geometry/channel fix when the
# old cache is invalid but rm is unavailable).
FORCE = bool(os.environ.get("DOSERAD_FORCE"))


def process_patient(pid: str) -> int:
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    out_dir = WORK / "cache" / "crops" / "photon" / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for beam in plan.beams:
        for cp in beam.control_points:
            out = out_dir / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if out.exists() and not FORCE:
                continue
            chans = photon_channels(
                image=ct, machine=MACHINE, iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm))
            dose = load_mha(pdir / "dose" /
                            f"Dose_B{beam.beam_idx}_CP{cp.cp_idx:03d}.mha").array
            bbox = compute_bbox(chans[2], margin=MARGIN)
            ch_c = crop_to_bbox(chans, bbox).astype(np.float16)
            ds_c = crop_to_bbox(dose, bbox).astype(np.float16)
            # atomic write (tmp + rename) so parallel workers can't corrupt the same file
            tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, channels=ch_c, dose=ds_c,
                                bbox=np.asarray(bbox, dtype=np.int32),
                                dose_max=np.float32(dose.max()))
            os.replace(tmp, out)
            n += 1
    return n


def main(pids):
    if not pids:
        pids = sorted(build_index(ROOT).patient_id.unique())
    # optional sharding for CPU-parallel workers: DOSERAD_SHARD="k/N" -> this worker
    # processes patients i where i % N == k (disjoint sets, no two workers share a patient).
    shard = os.environ.get("DOSERAD_SHARD")
    if shard:
        k, nshard = (int(x) for x in shard.split("/"))
        pids = [p for i, p in enumerate(pids) if i % nshard == k]
        print(f"[shard {k}/{nshard}] {len(pids)} patients", flush=True)
    for pid in pids:
        print(f"{pid}: wrote {process_patient(pid)} crop files", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
