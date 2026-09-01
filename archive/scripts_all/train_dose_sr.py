"""Tiny dose super-resolution net: 3mm dose (trilinear-upsampled) + 2mm density -> 2mm dose residual.

The 3mm low-res inference play outputs trilinear-upsampled dose; trilinear cannot restore penumbra
sharpness / peak (gamma 1%/1mm risk). This SR net learns the 3mm->2mm detail, GUIDED by the 2mm
density (available free at deploy from the source CT). Trained on GT pairs from the two caches
(photon_skinentry_3mm_m16 GT dose @3mm  vs  photon_skinentry_m24 GT dose @2mm, both 48mm-margin,
world-aligned via bbox*spacing with a shared origin).

Model: 5-layer 16ch residual CNN (receptive ~22mm). Normalization: per-CP by the 2mm crop dose max.
Loss: L1 with 3x weight where GT >= 10% of crop max.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/train_dose_sr.py --out RUNDIR [--steps 60000]
"""
from __future__ import annotations
import argparse, os, random, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_3mm_m16")
C2 = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
SPLITS = "/home/kaiwang/doserad2026_workdir/splits_all75.json"
SP3, SP2 = 3.0, 2.0


class SRNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(2, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(ch, 1, 3, padding=1))

    def forward(self, x):          # x: (B,2,z,y,x) = [upsampled dose (norm), density]
        return x[:, :1] + self.body(x)


def _upsample_world(d3, b3, b2, shape2, dev):
    """Trilinear-sample the 3mm crop onto the 2mm crop's voxels (shared origin, axis-aligned)."""
    z0, _, y0, _, x0, _ = b3
    Z0, _, Y0, _, X0, _ = b2
    dz, dy, dx = d3.shape
    t = torch.as_tensor(d3, dtype=torch.float32, device=dev)[None, None]
    ks = [((torch.arange(shape2[i], device=dev, dtype=torch.float32) + (Z0, Y0, X0)[i]) * SP2 / SP3)
          - (z0, y0, x0)[i] for i in range(3)]
    gz = (2 * ks[0] / max(dz - 1, 1)) - 1
    gy = (2 * ks[1] / max(dy - 1, 1)) - 1
    gx = (2 * ks[2] / max(dx - 1, 1)) - 1
    grid = torch.stack(torch.meshgrid(gz, gy, gx, indexing="ij"), dim=-1)[None][..., [2, 1, 0]]
    return F.grid_sample(t, grid, mode="bilinear", align_corners=True, padding_mode="border")[0, 0]


def load_pair(pid, f, dev):
    z3 = np.load(C3 / pid / f); z2 = np.load(C2 / pid / f)
    d3 = z3["dose"].astype(np.float32); b3 = z3["bbox"]
    d2 = z2["dose"].astype(np.float32); b2 = z2["bbox"]
    dens2 = z2["channels"][0].astype(np.float32)      # ch0 = density on the 2mm crop
    up = _upsample_world(d3, b3, b2, d2.shape, dev)
    m = float(d2.max())
    if m <= 0: return None
    return up / m, torch.as_tensor(dens2, device=dev), torch.as_tensor(d2, device=dev) / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/kaiwang/doserad2026_workdir/runs/sp3_srnet")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import json
    pids = json.load(open(SPLITS))["fold_all"]["train"]
    files = {p: sorted(f.name for f in (C3 / p).glob("*.npz") if ".tmp" not in f.name) for p in pids}
    net = SRNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    P = a.patch
    print(f"[sr] {len(pids)} pids, steps {a.steps}, patch {P}, batch {a.batch}", flush=True)

    cache, losses = {}, []
    for step in range(1, a.steps + 1):
        xb, yb = [], []
        while len(xb) < a.batch:
            pid = random.choice(pids); f = random.choice(files[pid])
            key = (pid, f)
            if key not in cache:
                if len(cache) > 400: cache.pop(next(iter(cache)))
                pair = load_pair(pid, f, dev)
                if pair is None: continue
                cache[key] = pair
            up, dens, gt = cache[key]
            zz, yy, xx = gt.shape
            if min(zz, yy, xx) < 8: continue
            pz, py, px = min(P, zz), min(P, yy), min(P, xx)
            z0 = random.randint(0, zz - pz); y0 = random.randint(0, yy - py); x0 = random.randint(0, xx - px)
            sl = (slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px))
            x = torch.stack([up[sl], dens[sl]])
            y = gt[sl][None]
            if pz < P or py < P or px < P:
                x = F.pad(x, (0, P - px, 0, P - py, 0, P - pz))
                y = F.pad(y, (0, P - px, 0, P - py, 0, P - pz))
            xb.append(x); yb.append(y)
        xb = torch.stack(xb); yb = torch.stack(yb)
        with torch.autocast("cuda"):
            pred = net(xb)
            w = 1.0 + 2.0 * (yb >= 0.1)
            loss = (w * (pred - yb).abs()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0:
            print(f"step {step}/{a.steps} | loss {np.mean(losses[-500:]):.5f} | lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % 10000 == 0 or step == a.steps:
            torch.save({"model": net.state_dict(), "step": step}, out / "sr.pt")
    print(f"done -> {out}/sr.pt", flush=True)


if __name__ == "__main__":
    main()
