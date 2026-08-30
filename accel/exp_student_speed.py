"""Go/no-go gate for distillation: is a smaller student net actually faster on the REAL photon crops?
Net time is partly memory/launch-bound so FLOP cuts don't fully translate. Measure base 48 (teacher)
vs 32 vs 24 forward ms on real photon CP crop shapes (compiled, warm).
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_student_speed.py 1ABB006
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.inference.pipeline import _build_coords
from doserad.eval.plan_predict import pad_to_multiple
from doserad.io.mha import load_mha
from accel.photon_channels_fast import photon_channels_fast

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"


def net(base):
    n = DoseUNet3D(in_ch=6, base=base, levels=4, bottleneck="dilated").to(DEV).eval()
    return torch.compile(n, dynamic=True)


def params(base):
    n = DoseUNet3D(in_ch=6, base=base, levels=4, bottleneck="dilated")
    return sum(p.numel() for p in n.parameters()) / 1e6


def main():
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    dn = hu_to_density(img.array, machine.hu_anchors); dt = torch.as_tensor(dn, dtype=torch.float32, device=DEV)
    coords = _build_coords(img, DEV); plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    mod = torch.zeros(1, dtype=torch.long, device=DEV)
    cps = [(b.get("iso_center", [0, 0, 0]), cp) for b in plan["beams"] for cp in b["control_points"]]
    shapes = []
    for iso, cp in cps[:30]:
        crop, _ = photon_channels_fast(img, machine, iso, cp["gantry_angle"], np.asarray(cp["mlc_left_int_mm"]),
                                       np.asarray(cp["mlc_right_int_mm"]), dens_t=dt, coords=coords, crop_margin=8)
        xp, _ = pad_to_multiple(crop.unsqueeze(0), factor=8)
        shapes.append(tuple(xp.shape[-3:]))
    reps = sorted(set(shapes))[::max(1, len(set(shapes))//5)][:6]

    nets = {b: net(b) for b in (48, 32, 24)}

    def tm(n, s, k=15):
        x = torch.randn(1, 6, *s, device=DEV)
        with torch.no_grad(), torch.autocast("cuda"):
            for _ in range(5): n(x, mod)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(k): n(x, mod)
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / k * 1000

    print("params (M):", {b: round(params(b), 1) for b in (48, 32, 24)})
    print(f"{'shape':>18} {'Mvox':>6} {'b48 ms':>8} {'b32 ms':>8} {'b24 ms':>8} {'32x':>6} {'24x':>6}")
    agg = {48: 0.0, 32: 0.0, 24: 0.0}
    for s in reps:
        t48, t32, t24 = tm(nets[48], s), tm(nets[32], s), tm(nets[24], s)
        agg[48] += t48; agg[32] += t32; agg[24] += t24
        print(f"{str(s):>18} {np.prod(s)/1e6:>6.2f} {t48:>8.2f} {t32:>8.2f} {t24:>8.2f} {t48/t32:>5.2f}x {t48/t24:>5.2f}x")
    print(f"\n  MEAN speedup vs base48:  base32 {agg[48]/agg[32]:.2f}x   base24 {agg[48]/agg[24]:.2f}x")


if __name__ == "__main__":
    main()
