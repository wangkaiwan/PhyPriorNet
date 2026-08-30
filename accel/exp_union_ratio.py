"""Decisive measurement for the 2-pass build lever: build_ray runs ONE grid_sample over the union of
a ray's beamlet geom-boxes. The 2-pass would rebuild over the union of the PB-TIGHT boxes. If a ray's
beamlets span the depth range, the tight union may barely shrink -> 2-pass wouldn't help. Measure, per
ray: vol(union geom boxes) vs vol(union tight boxes), and the achievable build speedup = geom/tight.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python accel/exp_union_ratio.py 1ABB006
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.io.mha import load_mha
from accel.proton_build_ray import build_ray
from accel.proton_ray_batch import _union_bbox
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb

DEV = "cuda"
ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACH = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
_vol = lambda bb: (bb[1]-bb[0]+1)*(bb[3]-bb[2]+1)*(bb[5]-bb[4]+1)


@torch.no_grad()
def main():
    pm = ProtonMachineData(device=DEV); machine = load_photon_machine(MACH)
    img = load_mha(f"{ROOT}/{PID}/image/ct.mha")
    density_np = hu_to_density(img.array, machine.hu_anchors); density_t = torch.as_tensor(density_np, device=DEV)
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    geom_vs, tight_vs, ratios, nray = [], [], [], 0
    for bi, b in enumerate(plan["beams"]):
        for ri, r in enumerate(b["rays"]):
            bls = []
            for l, bl in enumerate(r["beamlets"]):
                gb = geom_bbox_proton(density_np, img.spacing, img.origin, r["ray_source"],
                                      r["ray_target"], pm, bl["energy"])
                if gb is not None:
                    bls.append(dict(energy=bl["energy"], bbox=gb, key=(bi, ri, l)))
            if not bls:
                continue
            geom_union = _union_bbox([x["bbox"] for x in bls])
            stacks = build_ray(img, r["ray_source"], r["ray_target"], bls, machine=pm, density=density_t, device=DEV)
            tights = []
            for (stack, gbb), bl in zip(stacks, bls):
                pb = stack[2] * _P_CH_SCALE_PRIOR[2] / PROTON_DOSE_SCALE
                tb = _tight_from_pb(pb)
                if tb is None:
                    continue
                z0, z1, y0, y1, x0, x1 = tb; gz0, _, gy0, _, gx0, _ = gbb
                tights.append((gz0+z0, gz0+z1, gy0+y0, gy0+y1, gx0+x0, gx0+x1))
            if not tights:
                continue
            tight_union = _union_bbox(tights)
            gv, tv = _vol(geom_union), _vol(tight_union)
            geom_vs.append(gv); tight_vs.append(tv); ratios.append(gv/max(tv,1)); nray += 1
    r = np.asarray(ratios)
    print(f"\n=== RAY UNION SIZE ({PID}, {nray} rays) ===")
    print(f"  geom union vox: mean {np.mean(geom_vs):,.0f}   tight union vox: mean {np.mean(tight_vs):,.0f}")
    print(f"  geom/tight ratio (achievable pass-2 build speedup): "
          f"mean {r.mean():.2f}x  median {np.median(r):.2f}x  min {r.min():.2f}x  p10 {np.percentile(r,10):.2f}x")
    print(f"  -> pass-2 build ~= 63ms / {r.mean():.1f} = {63/r.mean():.1f}ms; "
          f"pass-1 coarse (stride2,step2) ~= {63/16:.1f}ms; 2-pass total ~= {63/r.mean()+63/16:.1f}ms vs 63ms")


if __name__ == "__main__":
    main()
