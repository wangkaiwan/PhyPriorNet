"""Test torch.compile(mode='reduce-overhead') (CUDA graphs) vs default compile on the dose nets.
Proton net on small tight crops is launch-latency-bound (~8ms overhead on 2-5ms compute) -> CUDA
graph replay should help. Photon net on large crops is compute-bound -> expect little. Measures
speed + output max|Δ| (lossless check). PARTICLE = proton|photon.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_reduce_overhead.py proton
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from doserad.model.unet3d import DoseUNet3D

DEV = "cuda"
PARTICLE = sys.argv[1] if len(sys.argv) > 1 else "proton"
if PARTICLE == "proton":
    W = "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt"; ICH = 5
    SHAPES = [(32, 48, 48), (32, 64, 64), (32, 80, 80), (48, 80, 80)]
else:
    W = "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt"; ICH = 6
    SHAPES = [(88, 248, 48), (88, 248, 96), (88, 248, 128)]


def build(mode=None):
    net = DoseUNet3D(in_ch=ICH, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    return torch.compile(net, dynamic=True, mode=mode) if mode else torch.compile(net, dynamic=True)


def tm(net, x, mod, n=20):
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(6): net(x, mod)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): net(x, mod)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    mod = torch.zeros(1, dtype=torch.long, device=DEV)
    net_d = build(); net_ro = build("reduce-overhead")
    print(f"=== {PARTICLE.upper()} net: default compile vs reduce-overhead (CUDA graphs) ===")
    print(f"{'shape':>16} {'default ms':>11} {'reduce-ov ms':>13} {'speedup':>8} {'max|Δ|':>10}")
    for s in SHAPES:
        x = torch.randn(1, ICH, *s, device=DEV)
        ta = tm(net_d, x, mod); tb = tm(net_ro, x, mod)
        with torch.no_grad(), torch.autocast("cuda"):
            ya = net_d(x, mod).float(); yb = net_ro(x, mod).float()
        md = (ya - yb).abs().max().item() / max(ya.abs().max().item(), 1e-9)
        print(f"{str(s):>16} {ta:>11.2f} {tb:>13.2f} {ta/tb:>7.2f}x {md:>10.2e}")


if __name__ == "__main__":
    main()
