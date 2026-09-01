"""Precompute beam-bbox-cropped channels + dose per CP, using the SKIN-ENTRY
radiological depth (experiment). Writes to crops/photon_skinentry_ssd/<pid>/.
Identical to precompute_photon_crops.py except photon_channels_skinentry.
DOSERAD_SHARD=k/N for CPU/GPU-parallel sharding; DOSERAD_FORCE=1 to overwrite."""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import os, sys
from pathlib import Path
import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.data.crop import compute_bbox, crop_to_bbox
from doserad.data.index import build_index
from doserad.io.mha import load_mha
from doserad.physics.channels_skinentry import photon_channels_skinentry
from doserad.physics.machine import load_photon_machine

ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training")
OUT_ROOT = Path(os.environ.get("DOSERAD_SKINENTRY_OUT",
                (os.environ.get("WORKDIR", "./workdir") + "/cache/crops/photon_skinentry_ssd")))
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")
MARGIN = 8
FORCE = bool(os.environ.get("DOSERAD_FORCE"))


def process_patient(pid: str) -> int:
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    out_dir = OUT_ROOT / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for beam in plan.beams:
        for cp in beam.control_points:
            out = out_dir / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if out.exists() and not FORCE:
                continue
            chans = photon_channels_skinentry(
                image=ct, machine=MACHINE, iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm))
            dose = load_mha(pdir / "dose" /
                            f"Dose_B{beam.beam_idx}_CP{cp.cp_idx:03d}.mha").array
            bbox = compute_bbox(chans[2], margin=MARGIN)
            ch_c = crop_to_bbox(chans, bbox).astype(np.float16)
            ds_c = crop_to_bbox(dose, bbox).astype(np.float16)
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
    shard = os.environ.get("DOSERAD_SHARD")
    if shard:
        k, nshard = (int(x) for x in shard.split("/"))
        pids = [p for i, p in enumerate(pids) if i % nshard == k]
        print(f"[shard {k}/{nshard}] {len(pids)} patients", flush=True)
    for pid in pids:
        print(f"{pid}: wrote {process_patient(pid)} crop files", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
