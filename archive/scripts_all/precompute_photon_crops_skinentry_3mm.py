"""3mm-RESOLUTION photon cache (the low-res inference play, user 2026-08-25 "先只训3mm").

Identical to precompute_photon_crops_skinentry.py except: CT and per-CP GT dose are first resampled
2mm -> 3mm (linear, world-aligned), channels are computed ON the 3mm grid (physics is mm-based so
photon_channels_skinentry is grid-agnostic), and the crop margin is the PHYSICAL equivalent of the
m24 training margin: 24 vox @2mm = 48mm -> 16 vox @3mm. Same npz format -> same trainer.

Rationale: photon runtime is 85% net forward on 4-6M-voxel 2mm crops; 3mm = 0.30x voxels -> ~2-3x
per-CP. Board sim says 80.3s -> ~40s = Mean ~10 = #1 IF gamma holds. Output is upsampled back to the
2mm source grid at deploy (dose is smooth; clinical grids are 2.5-3mm standard).

DOSERAD_SHARD=k/N sharding; DOSERAD_FORCE=1 overwrite.
"""
from __future__ import annotations

import os, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doserad.beam.parse import load_photon_plan
from doserad.data.crop import compute_bbox, crop_to_bbox
from doserad.data.index import build_index
from doserad.physics.channels_skinentry import photon_channels_skinentry
from doserad.physics.machine import load_photon_machine

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
OUT_ROOT = Path(os.environ.get("DOSERAD_SKINENTRY_OUT",
                "/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_3mm_m16"))
MACHINE = load_photon_machine(f"{ROOT}/beam_parameters.json")
MARGIN = 16              # 48mm physical == m24@2mm
SP = 3.0                 # target isotropic spacing (mm)
FORCE = bool(os.environ.get("DOSERAD_FORCE"))


class _Vol:
    """Volume-like wrapper (array z,y,x + spacing x,y,z + origin) as channels expects."""
    def __init__(self, img: sitk.Image):
        self.array = sitk.GetArrayFromImage(img)
        self.spacing = img.GetSpacing()
        self.origin = img.GetOrigin()


def _res3(img: sitk.Image, default: float) -> sitk.Image:
    sz = img.GetSize(); sp = img.GetSpacing()
    nsz = [int(round(sz[i] * sp[i] / SP)) for i in range(3)]
    return sitk.Resample(img, nsz, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                         (SP, SP, SP), img.GetDirection(), default, sitk.sitkFloat32)


def process_patient(pid: str) -> int:
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = _Vol(_res3(sitk.ReadImage(str(pdir / "image" / "ct.mha")), -1000.0))
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
            dose = sitk.GetArrayFromImage(_res3(
                sitk.ReadImage(str(pdir / "dose" / f"Dose_B{beam.beam_idx}_CP{cp.cp_idx:03d}.mha")), 0.0))
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
        k, N = (int(x) for x in shard.split("/"))
        pids = [p for i, p in enumerate(pids) if i % N == k]
    print(f"[3mm-cache] {len(pids)} patients -> {OUT_ROOT} (margin {MARGIN} vox = {MARGIN*SP:.0f}mm)", flush=True)
    for pid in pids:
        n = process_patient(pid)
        print(f"  {pid}: {n} CPs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
