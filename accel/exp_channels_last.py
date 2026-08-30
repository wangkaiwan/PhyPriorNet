"""Experiment: channels_last_3d memory format on the compiled proton dose net.

Attacks the biggest deploy component (~54ms net). channels_last is a memory-LAYOUT change (math is
the same modulo fp accumulation), so it should be ~gamma-lossless — verify both speed AND max|Δ|.
Compares, on representative proton tight-crop shapes (warm, steady-state):
  A) contiguous (current)   B) channels_last_3d
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_channels_last.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from doserad.model.unet3d import DoseUNet3D

DEV = "cuda"
W = "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt"
# representative proton tight crops (padded to /16); ~150-320k vox
SHAPES = [(32, 48, 48), (32, 64, 64), (48, 80, 80), (32, 96, 96)]


def _time(net, x, mod, n=30):
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(5):
            net(x, mod)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            net(x, mod)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    mod = torch.zeros(1, dtype=torch.long, device=DEV)

    cnet = torch.compile(net, dynamic=True)
    # channels_last variant: same weights, net converted to channels_last_3d
    net_cl = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    net_cl.load_state_dict(sd.get("ema", sd.get("model")))
    net_cl = net_cl.to(memory_format=torch.channels_last_3d)
    cnet_cl = torch.compile(net_cl, dynamic=True)

    print(f"{'shape':>16} {'vox':>8} {'contig ms':>10} {'chan_last ms':>13} {'speedup':>8} {'max|Δ|':>10}")
    for s in SHAPES:
        x = torch.randn(1, 5, *s, device=DEV)
        xcl = x.to(memory_format=torch.channels_last_3d)
        t_a = _time(cnet, x, mod)
        t_b = _time(cnet_cl, xcl, mod)
        with torch.no_grad(), torch.autocast("cuda"):
            ya = cnet(x, mod).float(); yb = cnet_cl(xcl, mod).float()
        md = (ya - yb).abs().max().item() / max(ya.abs().max().item(), 1e-9)
        vox = s[0] * s[1] * s[2]
        print(f"{str(s):>16} {vox:>8} {t_a:>10.2f} {t_b:>13.2f} {t_a/t_b:>7.2f}x {md:>10.2e}")


if __name__ == "__main__":
    main()
