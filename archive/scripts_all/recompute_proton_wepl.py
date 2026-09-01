"""TASK 3: recompute the proton WEPL channel CORRECTLY (per-voxel ray-march from the machine-SAD
source, validated in proton_pb_gpu) to replace the BEV-fan radiological_depth_fast WEPL that
over-counts ~30% on tilted gantries / 3mm-z. Writes a SEPARATE small cache (just the corrected WEPL
crop, like the prior cache); ProtonDoseDataset(wepl_dir=...) overrides channel 1 with it. Then retrain
no-prior to test if a physically-correct WEPL lifts it beyond 89.2. Does NOT touch the with-prior 96.1.

Shardable (GPU). Reuses proton_pb_gpu._wepl_crop / _compute_ssd / ProtonMachineData.
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
from doserad.physics.proton_pb_gpu import ProtonMachineData, _wepl_crop, _compute_ssd

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
NOPRIOR = "/home/kaiwang/doserad2026_workdir/cache/crops/proton"
OUT = "/home/kaiwang/doserad2026_workdir/cache/crops/proton_weplfix"


@torch.no_grad()
def _correct_wepl(image, density, ray_source, ray_target, bbox, pm, dev):
    sx, sy, sz = image.spacing; ox, oy, oz = image.origin
    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = tgt - jsrc; axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * pm.sad).astype(np.float32)          # machine-SAD source (matches pyRadPlan)
    z0, z1, y0, y1, x0, x1 = bbox
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)
    ssd = _compute_ssd(density, image.spacing, image.origin, src, axis.astype(np.float32), pm.sad, dev)
    wepl = _wepl_crop(density, image.spacing, image.origin, src, coords, dev,
                      march_start_mm=max(ssd - 50.0, 0.0))
    if isinstance(wepl, torch.Tensor):
        wepl = wepl.detach().cpu().numpy()
    return np.asarray(wepl, np.float32)


def process_patient(pid, pm, dev):
    npdir = Path(NOPRIOR) / pid
    if not npdir.exists():
        return 0
    odir = Path(OUT) / pid; odir.mkdir(parents=True, exist_ok=True)
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
    machine = load_photon_machine(MACHINE)
    density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
    rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]): (r["ray_source"], r["ray_target"])
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
    n = 0
    for f in sorted(x for x in npdir.glob("B*_R*_L*.npz") if ".tmp" not in x.name):
        out = odir / f.name
        if out.exists():
            n += 1; continue
        bb = tuple(int(v) for v in np.load(f)["bbox"])
        b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
        src, tgt = rays[(b, r, l)]
        wepl = _correct_wepl(ct, density, src, tgt, bb, pm, dev)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, wepl=wepl.astype(np.float16))
        tmp.replace(out); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pm = ProtonMachineData(device=dev)
    pids = sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    i, N = (int(x) for x in a.shard.split("/")); pids = pids[i::N]
    print(f"[wepl shard {i}/{N}] {len(pids)} patients on {dev}", flush=True)
    for k, pid in enumerate(pids):
        t = time.time()
        try:
            print(f"  [{k+1}/{len(pids)}] {pid}: {process_patient(pid, pm, dev)} [{time.time()-t:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k+1}/{len(pids)}] {pid}: ERROR {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
