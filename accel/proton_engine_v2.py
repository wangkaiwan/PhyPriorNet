"""Proton engine v2 — plan-level batched inference (exploration project, 2026-08-26).

Pipeline (vs the deployed per-beamlet loop in container/proton/predict.py):
  1. per BEAM: one orthographic BEV grid -> cumsum WEPL for all its beamlets   (388x on WEPL)
  2. per beamlet: PB kernel eval + 5ch stack + PB tight crop (vectorized ops)
  3. batched net forward: tight crops bucketed by pad16 shape, masked-GN batching
     (accel/masked_gn: padded-batch == per-sample EXACT)                        (amortizes launch)
Milestone ② verified the net is insensitive to the BEV WEPL swap (bev-vs-march plan gamma 99.66).

predict_plan_v2(image, beams, density_np, density_t, net, pm, dev) -> {key: (dose_np, bbox)} —
same contract as container.proton.predict.predict_beams.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from doserad.physics.proton_pb_gpu import _interp_1d, _compute_ssd
from doserad.physics.proton_pb_gpu_skinentry import _SKIN_THR
from accel.proton_build_ray import _E_SCALE, _SSD_DENSITY_THRESHOLD
from container.proton.geom_bbox import geom_bbox_proton
from container.proton.predict import _tight_from_pb, _r16

import importlib.util
_sp = importlib.util.spec_from_file_location("bw", str(Path(__file__).parent / "exp_batched_wepl.py"))
bw = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bw)
coords_of_bbox = bw.coords_of_bbox
wepl_beam_grid = bw.wepl_beam_grid
wepl_beam_sample = bw.wepl_beam_sample
import os
BEAMLET_CHUNK = int(os.environ.get("EV2_CHUNK", "6"))   # beamlets per streaming group (T4 memory)

MAX_BATCH_VOX = int(os.environ.get("EV2_MAXVOX", "14000000"))   # fwd batch cap in input voxels.
# base48 activations ~25x input: 14M vox ≈ 56MB input ≈ ~1.6GB activations — T4(14.5GB)-safe with
# density+BEV grids+jobs resident. (48M was 32GB-tuned and OOMed a real T4 job in group_norm.)
import os as _os
_PADQ = int(_os.environ.get("EV2_PADQ", "16"))     # bucket pad quantum (16=exact-min, 32=fewer buckets)
_rq = lambda v: -(-v // _PADQ) * _PADQ


@torch.no_grad()
def predict_plan_v2(image, beams, density_np, density_t, net, pm, dev="cuda", timings=None, on_frame=None):
    t = {"wepl": 0.0, "assemble": 0.0, "forward": 0.0}
    dens_t = density_t if torch.is_tensor(density_t) else torch.as_tensor(density_np, device=dev)
    dens5 = dens_t.view(1, 1, *dens_t.shape)
    pcs = torch.as_tensor(_P_CH_SCALE_PRIOR, device=dev).view(5, 1, 1, 1)
    out = {}
    jobs = []          # (key, tight_stack, full_bbox) — flushed per beam

    for b in beams:
        bi = b.get("beam_idx", beams.index(b))
        bd = np.asarray(b["rays"][0]["ray_target"], np.float64) - np.asarray(b["rays"][0]["ray_source"], np.float64)
        items = []     # (ri_idx, ray, l, bl, bb, coords)
        for ri, r in enumerate(b["rays"]):
            for l, bl in enumerate(r["beamlets"]):
                gb = geom_bbox_proton(density_np, image.spacing, image.origin,
                                      r["ray_source"], r["ray_target"], pm, bl["energy"])
                if gb is None:
                    continue
                items.append((ri, r, l, bl, gb, coords_of_bbox(gb, image.spacing, image.origin, dev)))
        if not items:
            continue
        # ---- v2c: per-beam BEV grid ONCE (fp16), then CHUNKED beamlet streaming (T4-safe) ----
        torch.cuda.synchronize(); t0 = time.time()
        corner_pts = []
        for it in items:
            z0, z1, y0, y1, x0, x1 = it[4]
            for cz in (z0, z1):
                for cy in (y0, y1):
                    for cx in (x0, x1):
                        corner_pts.append([cx * image.spacing[0] + image.origin[0],
                                           cy * image.spacing[1] + image.origin[1],
                                           cz * image.spacing[2] + image.origin[2]])
        probe = torch.tensor(corner_pts, dtype=torch.float32, device=dev)
        basis, ext, w_src_g, w_skin_g, ent_g = wepl_beam_grid(dens5, image.spacing, image.origin, bd,
                                                              coords_probe=probe)
        del probe
        torch.cuda.synchronize(); t["wepl"] += time.time() - t0

        ssd_cache = {}
        mod = torch.zeros(1, dtype=torch.long, device=dev)
        for c0 in range(0, len(items), BEAMLET_CHUNK):
            group = items[c0:c0 + BEAMLET_CHUNK]
            t0 = time.time()
            jobs = []
            for (ri, r, l, bl, bb, _unused) in group:
                coords = coords_of_bbox(bb, image.spacing, image.origin, dev)
                w_src, w_skin, entered = wepl_beam_sample(basis, ext, w_src_g, w_skin_g, ent_g, coords)
                if ri not in ssd_cache:
                    axis_f = np.asarray(r["ray_target"], np.float64) - np.asarray(r["ray_source"], np.float64)
                    axis_f = axis_f / np.linalg.norm(axis_f)
                    ssd_cache[ri] = (axis_f, _compute_ssd(dens_t, image.spacing, image.origin,
                                                          r["ray_source"], axis_f, pm.sad, dev,
                                                          threshold=_SSD_DENSITY_THRESHOLD))
                axis_f, ssd = ssd_cache[ri]
                z0, z1, y0, y1, x0, x1 = bb
                src_t = torch.as_tensor(r["ray_source"], device=dev, dtype=torch.float32)
                axis_t = torch.as_tensor(axis_f, device=dev, dtype=torch.float32)
                rel = coords - src_t
                along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
                lateral_dist = torch.linalg.norm(rel - along, dim=-1)
                del rel, along, coords
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
                stack = torch.stack([dens_u, w_src, pb * PROTON_DOSE_SCALE, lateral_dist,
                                     torch.full_like(lateral_dist, float(bl["energy"]) / _E_SCALE)], 0) / pcs
                del w_src, w_skin, entered, lateral_dist, pb, lateral, idd_v, sig_v, r2v, d, rad_depths_pb
                pbg = stack[2] * float(_P_CH_SCALE_PRIOR[2]) / PROTON_DOSE_SCALE
                tb = _tight_from_pb(pbg)
                if tb is None:
                    del stack; continue
                tz0, tz1, ty0, ty1, tx0, tx1 = tb
                tight = stack[:, tz0:tz1 + 1, ty0:ty1 + 1, tx0:tx1 + 1].half()
                del stack
                fb = (z0 + tz0, z0 + tz1, y0 + ty0, y0 + ty1, x0 + tx0, x0 + tx1)
                jobs.append(((bi, ri, l), tight, fb))
            t["assemble"] += time.time() - t0
            # bucketed forward within the group, emit immediately
            t0 = time.time()
            buckets = {}
            for j, (key, tight, fb) in enumerate(jobs):
                dd, hh, ww = tight.shape[-3:]
                buckets.setdefault((_rq(dd), _rq(hh), _rq(ww)), []).append(j)
            for pshape, idxs in buckets.items():
                xs, exts = [], []
                for j in idxs:
                    tight = jobs[j][1]
                    dd, hh, ww = tight.shape[-3:]
                    xs.append(F.pad(tight.float(), (0, pshape[2] - ww, 0, pshape[1] - hh, 0, pshape[0] - dd)))
                    exts.append((dd, hh, ww))
                x = torch.stack(xs)
                with torch.autocast("cuda"):
                    y = net(x, mod.expand(x.shape[0]))
                del x, xs
                for k, j in enumerate(idxs):
                    dd, hh, ww = exts[k]
                    dose = (y[k, 0, :dd, :hh, :ww].float() / PROTON_DOSE_SCALE).cpu().numpy().astype(np.float32)
                    if on_frame is not None:
                        on_frame(jobs[j][0], dose, jobs[j][2])
                    else:
                        out[jobs[j][0]] = (dose, jobs[j][2])
                del y
                torch.cuda.empty_cache()
            jobs = []
            t["forward"] += time.time() - t0
        del w_src_g, w_skin_g, ent_g
        torch.cuda.empty_cache()

    if timings is not None:
        timings.update(t)
    return out
