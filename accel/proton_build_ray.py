"""Proton ray-centric build — computes WEPL/SSD/lateral ONCE per ray and produces the full
5-channel input stack [density, WEPL, PB*scale, lateral, energy] for every beamlet of the ray.

Eliminates BOTH redundancies vs the deployed per-beamlet path:
  (a) sibling beamlets recompute identical geometry  -> compute once on the union bbox;
  (b) the ch1 WEPL channel and the PB-engine-internal WEPL are the same integral computed twice
      -> compute WEPL once, feed both the channel and the PB dose.
Byte-equivalent to the per-beamlet path (same _wepl_crop / _interp_1d math).

build_ray(image, ray, beamlets, machine, density) -> list of (input_stack_tensor[5,d,h,w], bbox)
in beamlet order, ready for the dose net (already /_P_CH_SCALE_PRIOR-scaled).
"""
from __future__ import annotations

import os
import numpy as np
import torch

from accel.wepl_fused import wepl_fused
from doserad.physics.proton_pb_gpu import (ProtonMachineData, _wepl_crop, _compute_ssd,
                                           _interp_1d, _SSD_DENSITY_THRESHOLD)
from doserad.physics.proton_pb_gpu_skinentry import _wepl_crop_skinentry, _SKIN_THR
from doserad.physics.proton_channels import _E_SCALE
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from accel.proton_ray_batch import _union_bbox


def build_ray(image, ray_source, ray_target, beamlets, *, machine: ProtonMachineData,
              density, device="cuda", wepl_step_mm: float = 1.0):
    # NOTE channels_last_3d on the compiled dose net is a DEAD END: proton net tiny (no benefit);
    # photon net 0.87-0.92x SLOWER on real large crops (inductor default layout already optimal).
    dev = device
    sx, sy, sz = image.spacing; ox, oy, oz = image.origin
    dens_is_tensor = torch.is_tensor(density)
    pcs = torch.as_tensor(_P_CH_SCALE_PRIOR, device=dev).view(5, 1, 1, 1)

    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = (tgt - jsrc); axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * machine.sad).astype(np.float32); axis_f = axis.astype(np.float32)

    ubb = _union_bbox([b["bbox"] for b in beamlets]); z0, z1, y0, y1, x0, x1 = ubb
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)

    ssd = _compute_ssd(density.detach() if dens_is_tensor else density, image.spacing,
                       image.origin, src, axis_f, machine.sad, dev, threshold=_SSD_DENSITY_THRESHOLD)
    # FUSED raymarch: one grid_sample -> source-ref WEPL (ch1) + skin-ref WEPL + entered (PB).
    # March from ssd-100 (the earlier start); the extra upstream air (rho~0) is ~0 to the source sum.
    wepl_ch, wepl_pb, entered = wepl_fused(density, image.spacing, image.origin, src, coords, dev,
                                           skin_thr=_SKIN_THR, march_start_mm=max(ssd - 100.0, 0.0),
                                           step_mm=wepl_step_mm)
    src_t = torch.as_tensor(src, device=dev); axis_t = torch.as_tensor(axis_f, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)
    dens_t = (density if dens_is_tensor else torch.as_tensor(density, device=dev, dtype=torch.float32))
    dens_u = dens_t[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    rad_depths_pb = wepl_pb * 10.0          # skin-referenced, for the PB depth lookup
    r2_full = lateral_dist ** 2

    out = []
    for bl in beamlets:
        eidx = machine.energy_index(bl["energy"]); n = int(machine.lengths[eidx].item())
        depths = machine.depths[eidx, :n]; idd = machine.conversion_factor * machine.idd[eidx, :n]
        sigma_d = machine.sigma[eidx, :n]; offset = float(machine.offset[eidx].item())
        sig_ini_sq = machine.sigma_ini(eidx, ssd) ** 2
        eff_depths = depths + offset                          # skin-referenced: NO rad_depth_offset
        bz0, bz1, by0, by1, bx0, bx1 = bl["bbox"]
        sl = (slice(bz0 - z0, bz1 - z0 + 1), slice(by0 - y0, by1 - y0 + 1), slice(bx0 - x0, bx1 - x0 + 1))
        d = rad_depths_pb[sl].reshape(-1)
        idd_v = _interp_1d(d, eff_depths, idd); sig_v = _interp_1d(d, eff_depths, sigma_d)
        oor = (d > eff_depths[-1]) | (d < eff_depths[0])
        idd_v = torch.where(oor, torch.zeros_like(idd_v), idd_v)
        r2v = r2_full[sl].reshape(-1)
        if os.environ.get("DOSERAD_LATERAL_DOUBLE") == "1" and machine.sigma1 is not None:
            # Hong DOUBLE-Gaussian lateral (nuclear-interaction halo): narrow core sigma1 + wide halo
            # sigma2, weight = fraction in the wide component (params already in the machine npz). Adds the
            # ~10% low-dose lateral halo the single effective `sigma` fit drops. OPT-IN; default unchanged.
            sig1_v = _interp_1d(d, eff_depths, machine.sigma1[eidx, :n])
            sig2_v = _interp_1d(d, eff_depths, machine.sigma2[eidx, :n])
            w_v = _interp_1d(d, eff_depths, machine.weight[eidx, :n]).clamp(0.0, 1.0)
            s1sq = sig1_v ** 2 + sig_ini_sq; s2sq = sig2_v ** 2 + sig_ini_sq
            g1 = torch.exp(-r2v / (2.0 * s1sq)) / (2.0 * torch.pi * s1sq)
            g2 = torch.exp(-r2v / (2.0 * s2sq)) / (2.0 * torch.pi * s2sq)
            lateral = (1.0 - w_v) * g1 + w_v * g2
        else:
            sigma_sq = sig_v ** 2 + sig_ini_sq
            lateral = torch.exp(-r2v / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
        pb = (lateral * idd_v).reshape(rad_depths_pb[sl].shape).clamp_min(0.0) * entered[sl]  # skin gate
        # 5-channel input stack (matches proton_dataset order), scaled
        lat_c = lateral_dist[sl]
        energy_c = torch.full_like(lat_c, float(bl["energy"]) / _E_SCALE)   # MeV/250, matches proton_channels
        stack = torch.stack([dens_u[sl], wepl_ch[sl], pb * PROTON_DOSE_SCALE, lat_c, energy_c], 0) / pcs
        out.append((stack, bl["bbox"]))
    return out
