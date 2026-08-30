"""Proton per-beamlet INFERENCE-speed benchmark vs the ≤1 s/dosemap gate (A10G).

Measures the REAL inference cost per beamlet (NOT the cache cost, which is dominated by GT-dose
disk loads + npz saves that don't happen at inference):
  (a) WEPL via radiological_depth_fast at FULL BEV res (128/128/256) vs REDUCED res — proton is a
      single narrow pencil, so a coarse BEV may match within tolerance for a big speedup.
  (b) one DoseUNet3D forward on the beamlet crop (random weights — fwd time is weight-independent).
  (c) total -> A10G projection (x1.5, the photon convention).

Image-level density is built ONCE per patient (amortized), like the photon runtime path.
Bbox taken from the cached no-prior npz (at submission it would come from the PB-prior extent).
Run AFTER the cache finishes (needs a free GPU):
    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/benchmark_proton_inference.py --pid 1ABB006 --n 24
"""
from __future__ import annotations
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch

from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine
from doserad.physics.density import hu_to_density
from doserad.physics.raytrace import radiological_depth_fast
from doserad.physics.proton_channels import _perp_basis, _E_SCALE
from doserad.model.unet3d import DoseUNet3D

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
CACHE = "/home/kaiwang/doserad2026_workdir/cache/crops/proton"
A10G = 1.5   # local->A10G scale (photon convention)


def _wepl(density, image, src, tgt, bbox, nu, nv, nd, dev):
    axis = (np.asarray(tgt) - np.asarray(src)); axis = (axis / (np.linalg.norm(axis) + 1e-9)).astype(np.float32)
    u, v = _perp_basis(axis)
    return radiological_depth_fast(density, image.spacing, image.origin, np.asarray(src, np.float32),
                                   axis, u, v, np.asarray(tgt, np.float32), n_u=nu, n_v=nv, n_d=nd,
                                   out_bbox=bbox, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="1ABB006")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--in-ch", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    machine = load_photon_machine(MACHINE)
    ct = load_mha(Path(ROOT) / a.pid / "image" / "ct.mha")
    plan = json.load(open(Path(ROOT) / a.pid / f"{a.pid}.json"))

    t = time.time()
    density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)   # ONCE per patient
    amortized = time.time() - t
    print(f"[amortized once/patient] density build {amortized*1000:.0f} ms", flush=True)

    # collect N beamlets (varied) with cached bbox
    cdir = Path(CACHE) / a.pid
    npzs = sorted(f for f in cdir.glob("B*_R*_L*.npz") if ".tmp" not in f.name)[: a.n * 6 : 6][: a.n]
    beams = {b["beam_idx"]: b for b in plan["beams"]}
    net = DoseUNet3D(in_ch=a.in_ch, base=48, levels=4, bottleneck="dilated").to(dev).eval()

    configs = {"full_128": (128, 128, 256), "red_64": (64, 64, 128), "red_48": (48, 48, 96)}
    times = {k: [] for k in configs}; fwd = []; corr = {k: [] for k in configs if k != "full_128"}
    @torch.no_grad()
    def run(npz):
        z = np.load(npz); bb = tuple(int(v) for v in z["bbox"])
        b, r, l = (int(npz.stem.split("_")[i][1:]) for i in range(3))
        beam = beams[b]; ray = next(rr for rr in beam["rays"] if rr["ray_idx"] == r)
        bl = next(bb2 for bb2 in ray["beamlets"] if bb2["beamlet_idx"] == l)
        wepls = {}
        for k, (nu, nv, nd) in configs.items():
            torch.cuda.synchronize() if dev == "cuda" else None; t0 = time.time()
            w = _wepl(density, ct, ray["ray_source"], ray["ray_target"], bb, nu, nv, nd, dev)
            torch.cuda.synchronize() if dev == "cuda" else None
            times[k].append(time.time() - t0); wepls[k] = w
        for k in corr:
            m = wepls["full_128"] > 0.1
            if m.sum() > 10:
                corr[k].append(float(np.corrcoef(wepls[k][m], wepls["full_128"][m])[0, 1]))
        # forward on the crop (assemble dummy channels of right shape)
        d, h, wid = wepls["full_128"].shape
        x = torch.randn(1, a.in_ch, d, h, wid, device=dev)
        pz, py, px = (-d) % 16, (-h) % 16, (-wid) % 16
        x = torch.nn.functional.pad(x, (0, px, 0, py, 0, pz))
        torch.cuda.synchronize() if dev == "cuda" else None; t0 = time.time()
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            net(x, torch.zeros(1, dtype=torch.long, device=dev))
        torch.cuda.synchronize() if dev == "cuda" else None
        fwd.append(time.time() - t0)

    for i, npz in enumerate(npzs):
        run(npz)
        if i == 0:  # warm-up discarded
            for k in times: times[k].clear()
            fwd.clear(); [corr[k].clear() for k in corr]
    print(f"\n=== per-beamlet (N={len(fwd)}, after warm-up) ===", flush=True)
    for k in configs:
        ms = np.median(times[k]) * 1000
        c = f" corr {np.mean(corr[k]):.4f}" if k in corr else " (ref)"
        print(f"  WEPL {k:8s}: {ms:6.1f} ms{c}", flush=True)
    fms = np.median(fwd) * 1000
    print(f"  forward         : {fms:6.1f} ms", flush=True)
    for k in configs:
        tot = np.median(times[k]) * 1000 + fms
        print(f"  TOTAL ({k}): local {tot:6.1f} ms | A10G ~{tot*A10G:6.1f} ms | gate 1000 ms -> "
              f"{'PASS' if tot*A10G < 1000 else 'FAIL'} ({1000/(tot*A10G):.1f}x)", flush=True)


if __name__ == "__main__":
    main()
