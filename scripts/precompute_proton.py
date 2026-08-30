"""Precompute per-beamlet proton crops (NO-PRIOR cache): channels [density, WEPL, lateral_dist,
energy] + GT dose, on a tight GT-derived bbox. NEW file; reuses proton_channels (which reuses the
photon ray-trace). PB prior is a SEPARATE cache (precompute_proton_prior.py, pyradplan env), loaded
by the dataset — mirrors the photon aaa_cache_dir split.

Shardable across GPUs:  CUDA_VISIBLE_DEVICES=0 python ... --shard 0/2   (and 1/2 on GPU1).
Per beamlet npz: channels(4,d,h,w) fp16, dose(d,h,w) fp16, bbox int32[6], energy f32, dose_max f32.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_channels import proton_channels

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
MARGIN = 4
THRESH = 0.01   # dose-region threshold (fraction of beamlet max) for the bbox


def _bbox_from_dose(dose, margin, shape):
    nz = np.argwhere(dose > THRESH * dose.max())
    if len(nz) == 0:
        return None
    z0, y0, x0 = nz.min(0); z1, y1, x1 = nz.max(0)
    return (max(int(z0) - margin, 0), min(int(z1) + margin, shape[0] - 1),
            max(int(y0) - margin, 0), min(int(y1) + margin, shape[1] - 1),
            max(int(x0) - margin, 0), min(int(x1) + margin, shape[2] - 1))


def process_patient(pid, out_root, machine, device):
    pdir = Path(ROOT) / pid
    ct = load_mha(pdir / "image" / "ct.mha")
    plan = json.load(open(pdir / f"{pid}.json"))
    odir = Path(out_root) / pid; odir.mkdir(parents=True, exist_ok=True)
    n = 0
    for beam in plan["beams"]:
        b = beam["beam_idx"]
        for ray in beam["rays"]:
            r = ray["ray_idx"]
            for bl in ray["beamlets"]:
                l = bl["beamlet_idx"]
                out = odir / f"B{b}_R{r}_L{l}.npz"
                if out.exists():
                    n += 1; continue
                dpath = pdir / "dose" / f"Dose_B{b}_R{r}_L{l}.mha"
                if not dpath.exists():
                    continue
                dose = sitk.GetArrayFromImage(sitk.ReadImage(str(dpath))).astype(np.float32)
                bbox = _bbox_from_dose(dose, MARGIN, dose.shape)
                if bbox is None:
                    continue
                chans, _ = proton_channels(ct, ray["ray_source"], ray["ray_target"],
                                           bl["energy"], machine.hu_anchors,
                                           out_bbox=bbox, device=device)
                z0, z1, y0, y1, x0, x1 = bbox
                ds_c = dose[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
                tmp = out.with_name(out.stem + ".tmp.npz")   # end in .npz so savez doesn't re-append
                np.savez_compressed(tmp, channels=chans.astype(np.float16),
                                    dose=ds_c.astype(np.float16),
                                    bbox=np.asarray(bbox, np.int32),
                                    energy=np.float32(bl["energy"]),
                                    dose_max=np.float32(dose.max()))
                tmp.replace(out)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/kaiwang/doserad2026_workdir/cache/crops/proton")
    ap.add_argument("--shard", default="0/1", help="i/N to process patient subset i of N")
    ap.add_argument("--pids", default=None, help="comma list to override")
    a = ap.parse_args()
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or __import__("torch").cuda.is_available() else "cpu"
    machine = load_photon_machine(MACHINE)
    pids = a.pids.split(",") if a.pids else sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    i, N = (int(x) for x in a.shard.split("/"))
    pids = pids[i::N]
    print(f"[shard {i}/{N}] {len(pids)} patients on {device}: {pids[:3]}...", flush=True)
    for k, pid in enumerate(pids):
        t = time.time()
        try:
            n = process_patient(pid, a.out_dir, machine, device)
            print(f"  [{k+1}/{len(pids)}] {pid}: {n} beamlets [{time.time()-t:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k+1}/{len(pids)}] {pid}: ERROR {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
