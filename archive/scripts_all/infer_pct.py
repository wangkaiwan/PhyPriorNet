"""Run the trained sCT model on each patient's MRI; write pseudo_ct.mha next
to ct.mha.

Usage: conda run -n doserad python scripts/infer_pct.py \
           --ckpt /home/kaiwang/doserad2026_workdir/runs/v1_sct/state.pt \
           [--pids 1ABB006 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from doserad.data.sct_dataset import _normalize_mri
from doserad.io.mha import Volume, load_mha, save_mha
from doserad.model.sct_unet import SCTUNet

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
SLICE_K = 2


def infer_one(net, mr_arr, device):
    """Slice-by-slice forward pass. mr_arr (z,y,x). Output HU (z,y,x)."""
    nz, ny, nx = mr_arr.shape
    mr_n = _normalize_mri(mr_arr)
    out = np.zeros_like(mr_arr, dtype=np.float32)
    # pad H,W to next multiple of 2^(levels-1)=8 for U-Net stride alignment
    pad_h = (-ny) % 8; pad_w = (-nx) % 8
    if pad_h or pad_w:
        mr_n = np.pad(mr_n, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    for z in range(nz):
        stack = np.zeros((1, 2 * SLICE_K + 1, mr_n.shape[1], mr_n.shape[2]),
                         dtype=np.float32)
        for j, zz in enumerate(range(z - SLICE_K, z + SLICE_K + 1)):
            if 0 <= zz < nz:
                stack[0, j] = mr_n[zz]
        x = torch.as_tensor(stack, device=device)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=(device == "cuda")):
                y = net(x).squeeze(0).squeeze(0).float().cpu().numpy()
        # un-pad back to native H,W
        out[z] = y[:ny, :nx]
    # de-standardize HU (training used CT/1000 clipped) -> multiply by 1000
    return (out * 1000.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pids", nargs="*", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SCTUNet(in_ch=2 * SLICE_K + 1, base=32, levels=4).to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(st["ema"])
    if args.pids:
        pids = args.pids
    else:
        pids = sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    for pid in pids:
        mr_path = Path(ROOT) / pid / "image" / "mr.mha"
        if not mr_path.exists():
            continue
        mr = load_mha(mr_path)
        pct = infer_one(net, mr.array, device)
        out = Volume(array=pct.astype(np.float32), spacing=mr.spacing,
                     origin=mr.origin, direction=mr.direction)
        save_mha(out, Path(ROOT) / pid / "image" / "pseudo_ct.mha")
        print(f"{pid}: wrote pseudo_ct.mha shape {pct.shape}", flush=True)


if __name__ == "__main__":
    main()
