"""Verify the 2-pass build (accel/proton_2pass.build_ray_2pass) is LOSSLESS vs the 1-pass deploy
(build_ray on geom box) and measure the build-time win. Runs the full deploy path (geom_bbox ->
build -> PB-tight -> compiled net), plan gamma vs MC GT + build ms/beamlet, for BASELINE and 2PASS.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_2pass.py 1ABB006
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.io.mha import load_mha
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from accel.proton_build_ray import build_ray
from accel.proton_2pass import build_ray_2pass
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
W = os.environ.get("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt")
MACH = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
CACHE = "/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
_r16 = lambda v: -(-v // 16) * 16


@torch.no_grad()
def run(img, dn, dt, beams, net, pm, mode):
    build_t = 0.0; out = {}
    for b in beams:
        for ri, r in enumerate(b["rays"]):
            bls = []
            for l, bl in enumerate(r["beamlets"]):
                gb = geom_bbox_proton(dn, img.spacing, img.origin, r["ray_source"], r["ray_target"], pm, bl["energy"])
                if gb is not None:
                    bls.append(dict(energy=bl["energy"], bbox=gb, key=(b["beam_idx"], ri, l)))
            if not bls:
                continue
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if mode == "2pass":
                stacks = build_ray_2pass(img, r["ray_source"], r["ray_target"], bls, machine=pm, density=dt, device=DEV)
            else:
                stacks = build_ray(img, r["ray_source"], r["ray_target"], bls, machine=pm, density=dt, device=DEV)
            torch.cuda.synchronize(); build_t += time.perf_counter() - t0
            for (stack, gbb), bl in zip(stacks, bls):
                pb = stack[2] * _P_CH_SCALE_PRIOR[2] / PROTON_DOSE_SCALE
                tb = _tight_from_pb(pb)
                if tb is None:
                    continue
                z0, z1, y0, y1, x0, x1 = tb
                tight = stack[:, z0:z1+1, y0:y1+1, x0:x1+1]
                d, h, w = tight.shape[-3:]
                x = F.pad(tight[None], (0, _r16(w)-w, 0, _r16(h)-h, 0, _r16(d)-d))
                with torch.autocast("cuda"):
                    y = net(x, torch.zeros(1, dtype=torch.long, device=DEV))
                dose = (y[0, 0, :d, :h, :w].float() / PROTON_DOSE_SCALE).cpu().numpy().astype(np.float32)
                gz0, _, gy0, _, gx0, _ = gbb
                out[bl["key"]] = (dose, (gz0+z0, gz0+z1, gy0+y0, gy0+y1, gx0+x0, gx0+x1))
    return out, build_t


def main():
    pm = ProtonMachineData(device=DEV); machine = load_photon_machine(MACH)
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load(W, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    net = torch.compile(net, dynamic=True)
    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    dn = hu_to_density(img.array, machine.hu_anchors); dt = torch.as_tensor(dn, device=DEV)
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
    full = img.array.shape
    gt_cps = [(np.load(f)["dose"].astype(np.float32), tuple(int(v) for v in np.load(f)["bbox"]))
              for f in sorted(g for g in Path(f"{CACHE}/{PID}").glob("B*_R*_L*.npz") if ".tmp" not in g.name)]
    gt = accumulate_plan(gt_cps, full); rx = float(gt.max())
    zz, yy, xx = np.where(gt >= 0.05*rx); m = 4
    crop = (slice(max(int(zz.min())-m,0),int(zz.max())+m+1), slice(max(int(yy.min())-m,0),int(yy.max())+m+1),
            slice(max(int(xx.min())-m,0),int(xx.max())+m+1))
    sp = img.spacing
    print(f"\n=== 2-PASS BUILD ({PID}) ===")
    base_plan = None
    for mode in ["baseline", "2pass"]:
        preds, bt = run(img, dn, dt, plan["beams"], net, pm, mode)
        pp = accumulate_plan([preds[k] for k in preds], full)
        g1c, g1m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=1.0, dta_mm=1.0)
        g1 = float((g1c[g1m] <= 1.0).mean())*100 if g1m.any() else float("nan")
        nb = len(preds)
        print(f"  {mode:>9}: gamma 1%/1mm {g1:>5.1f}%   build {bt/max(nb,1)*1000:>6.2f} ms/beamlet  ({nb} bl)")
        if mode == "baseline":
            base_plan = pp
        else:   # PLAN-level lossless check (proper normalization: rel to plan Rx)
            md = float(np.abs(pp - base_plan).max()) / max(rx, 1e-9)
            print(f"             lossless check: plan max|Δ| vs baseline = {md:.2e} (rel to Rx)")


if __name__ == "__main__":
    main()
