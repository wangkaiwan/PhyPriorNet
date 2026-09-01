"""Build the GPU-PB prior cache (option A): compute the analytical proton PB prior with the FAST GPU
engine (proton_pb_gpu, corr 0.995 vs pyRadPlan, ~40ms/beamlet) for every beamlet, on the SAME bbox as
the no-prior cache. Drop-in replacement for the pyRadPlan prior cache (same npz key "pb_prior", Gy).
Then finetune the with-prior model on THIS cache so train==inference (recovers the ~1.8 deployment gap).

Runs in `doserad` env on GPU (NOT pyradplan). Shardable. ~40ms/beamlet -> ~1h for 81000 on one GPU.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch

from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine
from doserad.physics.density import hu_to_density
from doserad.physics.proton_pb_gpu import ProtonMachineData, proton_pb_dose_gpu

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
NOPRIOR = "/home/kaiwang/doserad2026_workdir/cache/crops/proton"
OUT = "/home/kaiwang/doserad2026_workdir/cache/crops/proton_prior_gpu"


@torch.no_grad()
def process_patient(pid, pm, hu_anchors, dev, out_dir, mask_air=False):
    npdir = Path(NOPRIOR) / pid
    if not npdir.exists():
        return 0
    odir = Path(out_dir) / pid; odir.mkdir(parents=True, exist_ok=True)
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
    density = hu_to_density(ct.array, hu_anchors).astype(np.float32)         # once per patient
    body_mask = None
    if mask_air:   # external-air only: fill internal cavities so only outside-body gets zeroed
        from scipy.ndimage import binary_fill_holes
        body_mask = binary_fill_holes(density >= 0.1)
    plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
    rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]): (r["ray_source"], r["ray_target"], bl["energy"])
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
    n = 0
    for f in sorted(x for x in npdir.glob("B*_R*_L*.npz") if ".tmp" not in x.name):
        out = odir / f.name
        if out.exists():
            n += 1; continue
        bb = tuple(int(v) for v in np.load(f)["bbox"])
        b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
        src, tgt, e = rays[(b, r, l)]
        pb = proton_pb_dose_gpu(ct, src, tgt, e, out_bbox=bb, machine=pm,
                                density_override=density, device=dev,
                                mask_air=mask_air, body_mask=body_mask).astype(np.float32)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, pb_prior=pb.astype(np.float16))
        tmp.replace(out); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mask-air", action="store_true", help="zero external-air dose (body-mask)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pm = ProtonMachineData(device=dev)
    hu_anchors = load_photon_machine(MACHINE).hu_anchors
    pids = sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    i, N = (int(x) for x in a.shard.split("/")); pids = pids[i::N]
    print(f"[gpu-pb prior shard {i}/{N}] {len(pids)} patients on {dev}", flush=True)
    for k, pid in enumerate(pids):
        t = time.time()
        try:
            print(f"  [{k+1}/{len(pids)}] {pid}: {process_patient(pid, pm, hu_anchors, dev, a.out, a.mask_air)} [{time.time()-t:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k+1}/{len(pids)}] {pid}: ERROR {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
