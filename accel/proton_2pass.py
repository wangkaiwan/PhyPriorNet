"""2-pass proton build: the deployed build_ray computes the full 5-ch stack (expensive WEPL
grid_sample) over the inflated geom-union box, but the net only consumes the PB-TIGHT crop
(~4.5x smaller union). Pass 1 locates each beamlet's tight bbox with a CHEAP coarse PB (strided +
coarse-step WEPL); pass 2 runs the normal full-fidelity build_ray on those tight bboxes only.

LOSSLESS by construction: the net still sees the full-resolution build_ray channels on the same
PB-threshold crop it would have used before — only the WEPL is computed over a smaller box. Pass 1 is
a conservative SUPERSET locate (coarse threshold + margin) so it never clips the true tight crop.
"""
from __future__ import annotations

import numpy as np
import torch

from accel.wepl_fused import wepl_fused
from accel.proton_ray_batch import _union_bbox
from accel.proton_build_ray import build_ray
from doserad.physics.proton_pb_gpu import (ProtonMachineData, _compute_ssd, _interp_1d,
                                           _SSD_DENSITY_THRESHOLD)
from doserad.physics.proton_pb_gpu_skinentry import _SKIN_THR

_THRESH = 0.01     # matches precompute_proton / _tight_from_pb
_LOC_THRESH = 0.005  # conservative for the COARSE locate (never clip): half the deploy threshold


def _locate_tight(image, ray_source, ray_target, beamlets, *, machine: ProtonMachineData, density,
                  device="cuda", stride: int = 2, coarse_step: float = 2.0, margin: int = 4):
    """Cheap coarse PB over the geom union -> per-beamlet full-res tight bbox (superset). Returns a
    list aligned with `beamlets`: the tight bbox (z0,z1,y0,y1,x0,x1) clamped inside the geom bbox, or
    the geom bbox itself if the coarse PB is empty (safe fallback)."""
    dev = device
    sx, sy, sz = image.spacing; ox, oy, oz = image.origin
    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = (tgt - jsrc); axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * machine.sad).astype(np.float32); axis_f = axis.astype(np.float32)

    z0, z1, y0, y1, x0, x1 = _union_bbox([b["bbox"] for b in beamlets])
    # strided coarse grid over the geom union
    xs = ox + torch.arange(x0, x1 + 1, stride, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, stride, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, stride, device=dev, dtype=torch.float32) * sz
    nzc, nyc, nxc = len(zs), len(ys), len(xs)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)

    dens_is_t = torch.is_tensor(density)
    ssd = _compute_ssd(density.detach() if dens_is_t else density, image.spacing, image.origin,
                       src, axis_f, machine.sad, dev, threshold=_SSD_DENSITY_THRESHOLD)
    _, wepl_pb, entered = wepl_fused(density, image.spacing, image.origin, src, coords, dev,
                                     skin_thr=_SKIN_THR, march_start_mm=max(ssd - 100.0, 0.0),
                                     step_mm=coarse_step)
    src_t = torch.as_tensor(src, device=dev); axis_t = torch.as_tensor(axis_f, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)
    rad_depths = wepl_pb * 10.0
    r2 = lateral_dist ** 2

    out = []
    for bl in beamlets:
        bz0, bz1, by0, by1, bx0, bx1 = bl["bbox"]
        # geom bbox -> coarse-grid index range
        cz0, cz1 = (bz0 - z0) // stride, (bz1 - z0) // stride
        cy0, cy1 = (by0 - y0) // stride, (by1 - y0) // stride
        cx0, cx1 = (bx0 - x0) // stride, (bx1 - x0) // stride
        sl = (slice(cz0, cz1 + 1), slice(cy0, cy1 + 1), slice(cx0, cx1 + 1))
        eidx = machine.energy_index(bl["energy"]); n = int(machine.lengths[eidx].item())
        depths = machine.depths[eidx, :n]; idd = machine.conversion_factor * machine.idd[eidx, :n]
        sigma_d = machine.sigma[eidx, :n]; offset = float(machine.offset[eidx].item())
        sig_ini_sq = machine.sigma_ini(eidx, ssd) ** 2
        eff = depths + offset
        d = rad_depths[sl].reshape(-1)
        idd_v = _interp_1d(d, eff, idd); sig_v = _interp_1d(d, eff, sigma_d)
        oor = (d > eff[-1]) | (d < eff[0])
        idd_v = torch.where(oor, torch.zeros_like(idd_v), idd_v)
        ssq = sig_v ** 2 + sig_ini_sq
        lat = torch.exp(-r2[sl].reshape(-1) / (2.0 * ssq)) / (2.0 * torch.pi * ssq)
        pb = (lat * idd_v).reshape(rad_depths[sl].shape).clamp_min(0.0) * entered[sl]
        mx = float(pb.max())
        if mx <= 0:
            out.append(bl["bbox"]); continue
        nz = torch.nonzero(pb > _LOC_THRESH * mx, as_tuple=False)
        lo = nz.min(0).values.tolist(); hi = nz.max(0).values.tolist()
        # coarse idx -> full-res global, + margin, clamp to geom bbox
        tz0 = max(bz0 + lo[0] * stride - margin, bz0); tz1 = min(bz0 + hi[0] * stride + margin, bz1)
        ty0 = max(by0 + lo[1] * stride - margin, by0); ty1 = min(by0 + hi[1] * stride + margin, by1)
        tx0 = max(bx0 + lo[2] * stride - margin, bx0); tx1 = min(bx0 + hi[2] * stride + margin, bx1)
        out.append((tz0, tz1, ty0, ty1, tx0, tx1))
    return out


import os as _os
_MARGIN_DEFAULT = int(_os.environ.get("DOSERAD_2PASS_MARGIN", "8"))   # lossless (abd + lung), ~2.2x build


@torch.no_grad()
def build_ray_2pass(image, ray_source, ray_target, beamlets, *, machine: ProtonMachineData,
                    density, device="cuda", stride: int = 2, coarse_step: float = 2.0, margin: int = None):
    if margin is None:
        margin = _MARGIN_DEFAULT
    """Drop-in for build_ray but builds the full stack only over the PB-tight union. Returns
    [(stack[5,d,h,w], tight_bbox)] in beamlet order — stacks are ALREADY tight (net-ready)."""
    tights = _locate_tight(image, ray_source, ray_target, beamlets, machine=machine, density=density,
                           device=device, stride=stride, coarse_step=coarse_step, margin=margin)
    bl2 = [dict(energy=b["energy"], bbox=tb, key=b.get("key")) for b, tb in zip(beamlets, tights)]
    return build_ray(image, ray_source, ray_target, bl2, machine=machine, density=density, device=device)
