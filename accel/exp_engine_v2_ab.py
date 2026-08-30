"""Milestone ④: full A/B — deployed per-beamlet path vs proton engine v2 (batched) on one patient.
Reports: wall time (both), component split (v2), plan gamma v2-vs-deployed (equivalence), and
plan gamma vs MC GT for both (absolute). GT loading matches eval_proton_plan (z["dose"] raw).

Usage: CUDA_VISIBLE_DEVICES=1 python accel/exp_engine_v2_ab.py [pid]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.model.unet3d import DoseUNet3D
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma_gpu import gamma_array_gpu
from container.proton.predict import predict_beams
from accel.proton_engine_v2 import predict_plan_v2

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
DEV = "cuda"


class _Img:
    def __init__(self, arr, spacing, origin):
        self.array = arr; self.spacing = spacing; self.origin = origin


def plan_gamma(pp, ref, spacing, rx):
    zz, yy, xx = np.where(ref >= 0.05 * rx); m = 4
    cr = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1), slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
          slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
    g, k = gamma_array_gpu(pp[cr], ref[cr], spacing, rx, 1.0, 1.0, interp_fraction=10)
    return 100.0 * float((g[k] <= 1.0).mean()) if k.any() else float("nan")


@torch.no_grad()
def main():
    pdir = Path(ROOT) / PID
    ct = load_mha(pdir / "image" / "ct.mha")
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    pm = ProtonMachineData(device=DEV)
    dens = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    dens_t = torch.as_tensor(dens, device=DEV)
    plan = json.load(open(pdir / f"{PID}.json"))
    beams = [dict(b, beam_idx=bi) for bi, b in enumerate(plan["beams"])]
    import os as _o2
    net = DoseUNet3D(in_ch=5, base=int(_o2.environ.get("V2_BASE","48")), levels=4, bottleneck="dilated").to(DEV).eval()
    import os as _o
    sd = torch.load(_o.environ.get("V2_NET", "/home/kaiwang/doserad2026_workdir/runs/all75_r2_ft/state.pt"), map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))
    import os
    if os.environ.get("V2_COMPILE") == "1":
        net = torch.compile(net, dynamic=True)
    img = _Img(ct.array, ct.spacing, ct.origin)

    # deployed path (warm: run twice, time the 2nd)
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        dep = predict_beams(img, beams, dens, dens_t, net, pm, DEV)
        torch.cuda.synchronize(); t_dep = time.time() - t0
    torch.cuda.empty_cache()
    # engine v2 (warm)
    tt = {}
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        v2 = predict_plan_v2(img, beams, dens, dens_t, net, pm, DEV, timings=tt)
        torch.cuda.synchronize(); t_v2 = time.time() - t0
        torch.cuda.empty_cache()

    n = len(dep)
    print(f"{PID}: {n} beamlets", flush=True)
    print(f"deployed : {t_dep:.1f}s  ({t_dep*1000/n:.1f} ms/beamlet)", flush=True)
    print(f"engine v2: {t_v2:.1f}s  ({t_v2*1000/n:.1f} ms/beamlet)  speedup {t_dep/t_v2:.1f}x", flush=True)
    print(f"  v2 split: wepl {tt['wepl']:.2f}s | assemble {tt['assemble']:.2f}s | forward {tt['forward']:.2f}s", flush=True)

    full = dens.shape
    pp_dep = accumulate_plan(list(dep.values()), full)
    pp_v2 = accumulate_plan(list(v2.values()), full)
    gt_cps = []
    for f in sorted((CACHE / PID).glob("B*_R*_L*.npz")):
        if ".tmp" in f.name: continue
        z = np.load(f)
        gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max())
    rxd = float(pp_dep.max())
    print(f">>> equivalence v2-vs-deployed gamma: {plan_gamma(pp_v2, pp_dep, ct.spacing, rxd):.2f}", flush=True)
    print(f">>> vs GT: deployed {plan_gamma(pp_dep, gt, ct.spacing, rx):.2f}   v2 {plan_gamma(pp_v2, gt, ct.spacing, rx):.2f}", flush=True)


if __name__ == "__main__":
    main()
