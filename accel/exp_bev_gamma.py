"""Milestone ②: end-to-end gamma sensitivity of the EXISTING net to V3 (parallel-column BEV) WEPL.

For one patient: build the 5-channel stacks exactly like build_ray EXCEPT ch1/ch2 use the per-beam
orthographic BEV WEPL (36x faster, |Δ|~0.2 g/cm2 vs the per-voxel march). Run the deployed net on
both channel sets, accumulate plans, score gamma vs the MC GT plan (proton_ssd cache) AND deployed-
vs-V3 directly. Decides: (small drop) -> engine v2 can reuse the trained net; (collapse) -> the
engine needs a cache rebuild + retrain on its own physics.

Usage: CUDA_VISIBLE_DEVICES=1 python accel/exp_bev_gamma.py [pid]
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData, _interp_1d, _compute_ssd
from doserad.physics.proton_pb_gpu_skinentry import _SKIN_THR
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.model.unet3d import DoseUNet3D
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma_gpu import gamma_array_gpu
from accel.wepl_fused import wepl_fused
from accel.proton_build_ray import _E_SCALE, _SSD_DENSITY_THRESHOLD
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb, _r16

import importlib.util
_sp = importlib.util.spec_from_file_location("bw", str(Path(__file__).parent / "exp_batched_wepl.py"))
bw = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bw)

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
DEV = "cuda"
pcs = torch.as_tensor(_P_CH_SCALE_PRIOR, device=DEV).view(5, 1, 1, 1)


def assemble_and_predict(net, dens_t, ct, machine, pm, ray, beamlets_meta, wepls):
    """build_ray tail (verbatim math) with injected (w_src, w_skin, entered) per beamlet."""
    src = ray["ray_source"]; tgt = ray["ray_target"]
    axis_f = (np.asarray(tgt, np.float64) - np.asarray(src, np.float64))
    axis_f = axis_f / np.linalg.norm(axis_f)
    ssd = _compute_ssd(dens_t, ct.spacing, ct.origin, src, axis_f, pm.sad, DEV,
                       threshold=_SSD_DENSITY_THRESHOLD)
    src_t = torch.as_tensor(src, device=DEV, dtype=torch.float32)
    axis_t = torch.as_tensor(axis_f, device=DEV, dtype=torch.float32)
    out = []
    for (bl, bb, coords), (w_src, w_skin, entered) in zip(beamlets_meta, wepls):
        z0, z1, y0, y1, x0, x1 = bb
        rel = coords - src_t
        along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
        lateral_dist = torch.linalg.norm(rel - along, dim=-1)
        dens_u = dens_t[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        rad_depths_pb = w_skin * 10.0
        eidx = pm.energy_index(bl["energy"]); n = int(pm.lengths[eidx].item())
        depths = pm.depths[eidx, :n]; idd = pm.conversion_factor * pm.idd[eidx, :n]
        sigma_d = pm.sigma[eidx, :n]; offset = float(pm.offset[eidx].item())
        sig_ini_sq = pm.sigma_ini(eidx, ssd) ** 2
        eff_depths = depths + offset
        d = rad_depths_pb.reshape(-1)
        idd_v = _interp_1d(d, eff_depths, idd); sig_v = _interp_1d(d, eff_depths, sigma_d)
        oor = (d > eff_depths[-1]) | (d < eff_depths[0])
        idd_v = torch.where(oor, torch.zeros_like(idd_v), idd_v)
        r2v = (lateral_dist ** 2).reshape(-1)
        sigma_sq = sig_v ** 2 + sig_ini_sq
        lateral = torch.exp(-r2v / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
        pb = (lateral * idd_v).reshape(rad_depths_pb.shape).clamp_min(0.0) * entered
        lat_c = lateral_dist
        energy_c = torch.full_like(lat_c, float(bl["energy"]) / _E_SCALE)
        stack = torch.stack([dens_u, w_src, pb * PROTON_DOSE_SCALE, lat_c, energy_c], 0) / pcs
        # tight crop + net (predict.py verbatim)
        pbg = stack[2] * float(_P_CH_SCALE_PRIOR[2]) / PROTON_DOSE_SCALE
        tb = _tight_from_pb(pbg)
        if tb is None: continue
        tz0, tz1, ty0, ty1, tx0, tx1 = tb
        tight = stack[:, tz0:tz1 + 1, ty0:ty1 + 1, tx0:tx1 + 1]
        dd, hh, ww = tight.shape[-3:]
        x = F.pad(tight[None], (0, _r16(ww) - ww, 0, _r16(hh) - hh, 0, _r16(dd) - dd))
        with torch.autocast("cuda"):
            y = net(x, torch.zeros(1, dtype=torch.long, device=DEV))
        dose = (y[0, 0, :dd, :hh, :ww].float() / PROTON_DOSE_SCALE).cpu().numpy().astype(np.float32)
        fb = (z0 + tz0, z0 + tz1, y0 + ty0, y0 + ty1, x0 + tx0, x0 + tx1)
        out.append((dose, fb))
    return out


@torch.no_grad()
def main():
    pdir = Path(ROOT) / PID
    ct = load_mha(pdir / "image" / "ct.mha")
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    pm = ProtonMachineData(device=DEV)
    dens = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    dens_t = torch.as_tensor(dens, device=DEV)
    dens5 = dens_t.view(1, 1, *dens.shape)
    plan = json.load(open(pdir / f"{PID}.json"))
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    sd = torch.load("/home/kaiwang/doserad2026_workdir/runs/all75_r2_ft/state.pt", map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))

    variants = {"march": [], "bev": []}
    t_wepl = {"march": 0.0, "bev": 0.0}
    for b in plan["beams"]:
        bd = np.array(b["rays"][0]["ray_target"]) - np.array(b["rays"][0]["ray_source"])
        # collect all beamlets of the beam
        beam_items = []       # (ray, bl, bb, coords)
        for r in b["rays"]:
            for bl in r["beamlets"]:
                gb = geom_bbox_proton(dens, ct.spacing, ct.origin, r["ray_source"], r["ray_target"], pm, bl["energy"])
                if gb is None: continue
                beam_items.append((r, bl, gb, bw.coords_of_bbox(gb, ct.spacing, ct.origin, DEV)))
        if not beam_items: continue
        # V3 WEPL for the whole beam
        torch.cuda.synchronize(); t0 = time.time()
        v3 = bw.wepl_beam_ortho(dens5, ct.spacing, ct.origin, bd, [c for _, _, _, c in beam_items])
        torch.cuda.synchronize(); t_wepl["bev"] += time.time() - t0
        # march WEPL per beamlet
        torch.cuda.synchronize(); t0 = time.time()
        mm = [wepl_fused(dens_t, ct.spacing, ct.origin, ri["ray_source"], c, DEV, skin_thr=_SKIN_THR)
              for ri, _, _, c in beam_items]
        torch.cuda.synchronize(); t_wepl["march"] += time.time() - t0
        # group by ray and predict
        for name, weps in (("march", mm), ("bev", v3)):
            by_ray = {}
            for (ri, bl, bb, c), w in zip(beam_items, weps):
                by_ray.setdefault(id(ri), (ri, []))[1].append(((bl, bb, c), w))
            for _, (ri, lst) in by_ray.items():
                preds = assemble_and_predict(net, dens_t, ct, machine, pm, ri,
                                             [m for m, _ in lst], [w for _, w in lst])
                variants[name].extend(preds)

    # GT plan from cache
    gt_cps = []
    for f in sorted((CACHE / PID).glob("B*_R*_L*.npz")):
        z = np.load(f)
        gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    full = dens.shape
    gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max())
    res = {}
    for name, preds in variants.items():
        pp = accumulate_plan(preds, full)
        zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
        cr = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1), slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
              slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
        g, msk = gamma_array_gpu(pp[cr], gt[cr], ct.spacing, rx, 1.0, 1.0, interp_fraction=10)
        res[name] = (100.0 * float((g[msk] <= 1.0).mean()) if msk.any() else float("nan"), pp)
    print(f"\n{PID}: WEPL time march {t_wepl['march']:.1f}s vs bev {t_wepl['bev']:.1f}s "
          f"({t_wepl['march']/max(t_wepl['bev'],1e-9):.1f}x)", flush=True)
    print(f">>> plan gamma vs GT:  march {res['march'][0]:.2f}   bev {res['bev'][0]:.2f}   "
          f"Δ {res['bev'][0]-res['march'][0]:+.2f}", flush=True)
    # direct bev-vs-march plan gamma (march as reference) — GT-independent sensitivity readout
    ppm = res['march'][1]; ppb = res['bev'][1]
    rxm = float(ppm.max())
    zz, yy, xx = np.where(ppm >= 0.05 * rxm); m = 4
    cr2 = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1), slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
           slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
    g2, k2 = gamma_array_gpu(ppb[cr2], ppm[cr2], ct.spacing, rxm, 1.0, 1.0, interp_fraction=10)
    print(f">>> bev-vs-march direct gamma: {100.0*float((g2[k2]<=1.0).mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
