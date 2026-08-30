"""Quick MRI vs sCT vs CT image comparison (NO dose/gamma, CPU only) — to eyeball what's
wrong with the v1_sct sCT. Cols: MRI | sCT (v1_sct) | CT (real) | sCT−CT (HU)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doserad.data.sct_dataset import _normalize_mri
from doserad.io.mha import load_mha
from doserad.model.sct_unet import SCTUNet

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
SCT_CKPT = "/home/kaiwang/doserad2026_workdir/runs/v1_sct/state.pt"


def sct_infer(net, mr_arr, dev):
    nz, ny, nx = mr_arr.shape
    n = _normalize_mri(mr_arr)
    ph, pw = (-ny) % 8, (-nx) % 8
    if ph or pw:
        n = np.pad(n, ((0, 0), (0, ph), (0, pw)))
    o = np.zeros((nz, ny, nx), np.float32)
    for z in range(nz):
        s = np.zeros((1, 5, n.shape[1], n.shape[2]), np.float32)
        for j, zz in enumerate(range(z - 2, z + 3)):
            if 0 <= zz < nz:
                s[0, j] = n[zz]
        with torch.no_grad():
            y = net(torch.as_tensor(s, device=dev)).squeeze().float().cpu().numpy()
        o[z] = y[:ny, :nx]
    return (o * 1000.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="*", default=["1ABB006", "1ABB041", "1THB016", "1THB121"])
    ap.add_argument("--out-dir", default="/home/kaiwang/doserad2026_workdir/runs/sct_image_viz")
    args = ap.parse_args()
    dev = "cpu"
    net = SCTUNet(in_ch=5, base=32, levels=4).to(dev).eval()
    net.load_state_dict(torch.load(SCT_CKPT, map_location=dev)["ema"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["MRI", "sCT (v1_sct)", "CT (real)", "sCT − CT (HU)"]
    for pid in args.pids:
        mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha").array
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha").array
        sct = sct_infer(net, mr, dev)
        body = (ct > -500).sum(axis=(1, 2)); zc = int(np.argmax(body))
        zs = sorted({z for z in [int(zc * 0.7), zc, min(int(zc * 1.3), ct.shape[0] - 1)]
                     if 0 <= z < ct.shape[0]})
        fig, axs = plt.subplots(len(zs), 4, figsize=(16, 4 * len(zs)), squeeze=False)
        for ri, z in enumerate(zs):
            for ci in range(4):
                ax = axs[ri][ci]
                if ci == 0:
                    ax.imshow(mr[z], cmap="gray")
                elif ci == 1:
                    ax.imshow(sct[z], cmap="gray", vmin=-1000, vmax=1000)
                elif ci == 2:
                    ax.imshow(ct[z], cmap="gray", vmin=-1000, vmax=1000)
                else:
                    bd = ct[z] > -500
                    im = ax.imshow(np.where(bd, sct[z] - ct[z], np.nan), cmap="bwr",
                                   vmin=-400, vmax=400); plt.colorbar(im, ax=ax, fraction=.046)
                if ri == 0:
                    ax.set_title(cols[ci], fontsize=11)
                if ci == 0:
                    ax.set_ylabel(f"z={z}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        mae = float(np.abs(sct[ct > -500] - ct[ct > -500]).mean())
        fig.suptitle(f"{pid}  |  in-body HU MAE = {mae:.0f}  (sCT/CT shown in window [-1000,1000] HU)",
                     fontsize=12)
        fig.tight_layout(); out = out_dir / f"sct_img_{pid}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"{pid}: in-body HU MAE {mae:.0f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
