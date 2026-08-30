"""Analyze the photon net (now the photon bottleneck, ~62ms/CP). Is it real compute or overhead?
Measure: typical photon CP crop volume, pure net forward time (synced) at those shapes, and the
channels_last_3d speedup + output delta. Decides whether channels_last / other net levers are worth it.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_photon_net.py 1ABB006
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
W = "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"


def build_net(cl=False):
    net = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    if cl:
        net = net.to(memory_format=torch.channels_last_3d)
    return torch.compile(net, dynamic=True)


def main():
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    dn = hu_to_density(img.array, machine.hu_anchors); dt = torch.as_tensor(dn, dtype=torch.float32, device=DEV)
    coords = _build_coords(img, DEV); plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    mod = torch.zeros(1, dtype=torch.long, device=DEV)

    # collect real padded crop shapes (in_ch=6 -> add a naive channel placeholder)
    shapes, vols = [], []
    cps = [(b.get("iso_center", [0, 0, 0]), cp) for b in plan["beams"] for cp in b["control_points"]]
    for iso, cp in cps[:40]:
        crop, bbox = photon_channels_fast(img, machine, iso, cp["gantry_angle"],
                                          np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                                          dens_t=dt, coords=coords, crop_margin=8)
        x = torch.zeros(1, 6, *crop.shape[-3:], device=DEV)
        xp, _ = pad_to_multiple(x, factor=8)
        shapes.append(tuple(xp.shape[-3:])); vols.append(int(np.prod(xp.shape[-3:])))
    print(f"photon CP crop (padded) volume: mean {np.mean(vols)/1e6:.2f}M  min {min(vols)/1e6:.2f}M  max {max(vols)/1e6:.2f}M vox")

    net = build_net(cl=False); net_cl = build_net(cl=True)
    reps = shapes[:8]

    def tm(n, x, cl):
        xx = x.to(memory_format=torch.channels_last_3d) if cl else x
        with torch.no_grad(), torch.autocast("cuda"):
            for _ in range(3): n(xx, mod)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): n(xx, mod)
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / 10 * 1000

    print(f"{'shape':>18} {'Mvox':>6} {'net ms':>8} {'net_cl ms':>10} {'x':>6} {'max|Δ|':>9}")
    for s in reps:
        x = torch.randn(1, 6, *s, device=DEV)
        ta = tm(net, x, False); tb = tm(net_cl, x, True)
        with torch.no_grad(), torch.autocast("cuda"):
            ya = net(x, mod).float(); yb = net_cl(x.to(memory_format=torch.channels_last_3d), mod).float()
        md = (ya - yb).abs().max().item() / max(ya.abs().max().item(), 1e-9)
        print(f"{str(s):>18} {np.prod(s)/1e6:>6.2f} {ta:>8.2f} {tb:>10.2f} {ta/tb:>5.2f}x {md:>9.2e}")


if __name__ == "__main__":
    main()
