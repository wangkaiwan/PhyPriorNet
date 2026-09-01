"""Precompute the sCT-derived physics channels per CP for the Task-2 exp3 experiments.

For each control point we recompute photon_channels() from the **v4 synthetic CT** (treating sCT as
CT — the honest test-time setup, no GT-CT leakage), crop to the SAME bbox as the real-CT crop cache
(so dose/geometry line up), and store only the image-dependent channels [density, rdepth, fluence].
Geometry (dist_to_cax, source_dist) and dose are identical to the real cache and are NOT duplicated.

Out: $DOSERAD_WORK/cache/crops/photon_sct_v4/<pid>/<B>_<CP>.npz  with key 'sct_phys' (3,d,h,w) fp16.
Reuses the real cache bbox at cache/crops/photon/<pid>/<B>_<CP>.npz.
Sharding: DOSERAD_SHARD="k/N". Overwrite: DOSERAD_FORCE=1.
Usage: CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/precompute_photon_crops_sct.py [pid ...]
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.data.crop import crop_to_bbox
from doserad.data.index import build_index
from doserad.io.mha import load_mha
from doserad.physics.channels import photon_channels
from doserad.physics.machine import load_photon_machine

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
SCT_DIR = Path("/data/kwang/sct_dose/v4_sct")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
REAL_CACHE = WORK / "cache" / "crops" / "photon"
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")
FORCE = bool(os.environ.get("DOSERAD_FORCE"))


def process_patient(pid: str) -> int:
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    sct = load_mha(SCT_DIR / pid / "sCT.mha")          # sCT on the shared CT/dose grid
    out_dir = WORK / "cache" / "crops" / "photon_sct_v4" / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for beam in plan.beams:
        for cp in beam.control_points:
            name = f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            out = out_dir / name
            if out.exists() and not FORCE:
                continue
            real = REAL_CACHE / pid / name
            if not real.exists():
                continue                                # only CPs present in the real cache
            bbox = tuple(int(v) for v in np.load(real)["bbox"])
            chans = photon_channels(
                image=sct, machine=MACHINE, iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm))
            phys = crop_to_bbox(chans, bbox)[:3].astype(np.float16)   # density, rdepth, fluence (sCT)
            tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, sct_phys=phys)
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
        print(f"{pid}: wrote {process_patient(pid)} sct-phys crops", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
