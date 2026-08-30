"""End-to-end benefit of masked-GN batching for photon-CT: per-CP forward speedup and
plan-level accuracy vs the deployed per-sample (autocast, minimal self-pad) reference.

Times ONLY the dose-net forward (build is identical for both). Reference = current deploy
path (each CP padded to /16, batch 1). Test = pad each CP to /16, group into padded batches
under a voxel cap, forward under valid_extents. Accumulates both plans and reports the plan
max rel-diff (must be < 0.1% of plan max).

Run (GPU1): CUDA_VISIBLE_DEVICES=1 conda run -n doserad python -u accel/bench_masked_batch.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doserad.data.dataset import normalize_channels, DOSE_SCALE  # noqa: E402
from doserad.model.unet3d import DoseUNet3D  # noqa: E402
from accel.masked_gn import install_masked_batching, valid_extents  # noqa: E402

DEV = "cuda"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_ssd/1ABB006")
CAP = 6_000_000   # padded voxels/batch (masked-GN is exact, so we can push the cap up)


def r16(v):
    return -(-v // 16) * 16


def main():
    net = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    net.load_state_dict(torch.load(
        "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt",
        map_location=DEV)["ema"])
    files = sorted(CACHE.glob("*.npz"))
    crops, bbs, full = [], [], None
    for f in files:
        z = np.load(f)
        inp = normalize_channels(z["channels"].astype(np.float32), add_naive=True, naive_skin_gate=True)
        crops.append(torch.from_numpy(inp)); bbs.append(tuple(int(v) for v in z["bbox"]))
    print(f"{len(crops)} CPs")

    # infer full-volume shape from max bbox
    Z = max(b[1] for b in bbs) + 1; Y = max(b[3] for b in bbs) + 1; X = max(b[5] for b in bbs) + 1
    plan_ref = np.zeros((Z, Y, X), np.float32); plan_bat = np.zeros((Z, Y, X), np.float32)

    def paste(plan, y, bb):
        z0, z1, y0, y1, x0, x1 = bb
        plan[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += y

    # ---- reference: per-sample, pad/16, autocast ----
    torch.cuda.synchronize(); t0 = time.time()
    for c, bb in zip(crops, bbs):
        d, h, w = c.shape[-3:]
        x = F.pad(c[None].to(DEV), (0, r16(w) - w, 0, r16(h) - h, 0, r16(d) - d))
        with torch.no_grad(), torch.autocast("cuda"):
            y = net(x, torch.zeros(1, dtype=torch.long, device=DEV))
        paste(plan_ref, (y[0, 0, :d, :h, :w].float() / DOSE_SCALE).cpu().numpy(), bb)
    torch.cuda.synchronize(); t_ref = time.time() - t0

    # ---- test: masked-GN padded batches ----
    install_masked_batching(net)
    torch.cuda.synchronize(); t0 = time.time()
    buf = []   # (crop, bb, (d,h,w))
    def flush(buf):
        if not buf:
            return
        D = max(r16(c.shape[-3]) for c, _, _ in buf)
        H = max(r16(c.shape[-2]) for c, _, _ in buf)
        W = max(r16(c.shape[-1]) for c, _, _ in buf)
        xb = torch.zeros((len(buf), buf[0][0].shape[0], D, H, W), device=DEV)
        ext = []
        for i, (c, _, (d, h, w)) in enumerate(buf):
            xb[i, :, :d, :h, :w] = c.to(DEV); ext.append((r16(d), r16(h), r16(w)))
        with valid_extents(ext, full=(D, H, W)):
            with torch.no_grad(), torch.autocast("cuda"):
                yb = net(xb, torch.zeros(len(buf), dtype=torch.long, device=DEV))
        for i, (c, bb, (d, h, w)) in enumerate(buf):
            paste(plan_bat, (yb[i, 0, :d, :h, :w].float() / DOSE_SCALE).cpu().numpy(), bb)
    for c, bb in zip(crops, bbs):
        d, h, w = c.shape[-3:]
        nb = buf + [(c, bb, (d, h, w))]
        D = max(r16(x[0].shape[-3]) for x in nb); H = max(r16(x[0].shape[-2]) for x in nb); Wd = max(r16(x[0].shape[-1]) for x in nb)
        if buf and len(nb) * D * H * Wd > CAP:
            flush(buf); buf = []
        buf.append((c, bb, (d, h, w)))
    flush(buf)
    torch.cuda.synchronize(); t_bat = time.time() - t0

    pm = plan_ref.max()
    rel = np.abs(plan_bat - plan_ref).max() / max(pm, 1e-9)
    print(f"\nforward per-CP:  ref {t_ref/len(crops)*1e3:.1f} ms  batch {t_bat/len(crops)*1e3:.1f} ms"
          f"  -> {t_ref/t_bat:.2f}x")
    print(f"plan max rel-diff: {rel*100:.4f}%   (<0.1% = {'PASS' if rel < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    main()
