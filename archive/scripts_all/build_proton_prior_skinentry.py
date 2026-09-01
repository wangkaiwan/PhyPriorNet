"""Build the SKIN-ENTRY GPU-PB prior cache: per-beamlet proton PB dose computed with the
per-ray skin-entry engine (proton_pb_gpu_skinentry) — depth referenced from the skin, no
dose painted in the upstream air gap, Bragg peak physically placed (verified: lower dose-MAE
and peak error vs MC than the from-source prior; and, unlike mask_air, it actually moves the
peak). Same npz layout ("pb_prior", Gy fp16) and same bboxes as the no-prior cache, so it is a
drop-in `prior_dir` for training. Shardable GPU. ~similar cost to build_proton_prior_gpu.py.
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
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
NOPRIOR = "/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd"
OUT = "/home/kaiwang/doserad2026_workdir/cache/crops/proton_prior_skinentry_ssd"


@torch.no_grad()
def process_patient(pid, pm, hu_anchors, dev, out_dir):
    npdir = Path(NOPRIOR) / pid
    if not npdir.exists():
        return 0
    odir = Path(out_dir) / pid; odir.mkdir(parents=True, exist_ok=True)
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
    density = hu_to_density(ct.array, hu_anchors).astype(np.float32)
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
        pb = proton_pb_dose_gpu_skinentry(ct, src, tgt, e, out_bbox=bb, machine=pm,
                                          density_override=density, device=dev).astype(np.float32)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, pb_prior=pb.astype(np.float16))
        tmp.replace(out); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pm = ProtonMachineData(device=dev)
    hu_anchors = load_photon_machine(MACHINE).hu_anchors
    pids = sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    i, N = (int(x) for x in a.shard.split("/")); pids = pids[i::N]
    print(f"[skinentry-prior shard {i}/{N}] {len(pids)} patients on {dev}", flush=True)
    for k, pid in enumerate(pids):
        t = time.time()
        try:
            print(f"  [{k+1}/{len(pids)}] {pid}: {process_patient(pid, pm, hu_anchors, dev, a.out)} [{time.time()-t:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k+1}/{len(pids)}] {pid}: ERROR {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
