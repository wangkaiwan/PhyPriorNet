"""Proton Hong PB dose with PER-RAY SKIN-ENTRY depth referencing (experiment).

Motivation (user, 2026-07-08): the `mask_air` post-hoc body-mask multiply zeroes the
external-air dose but CANNOT move the Bragg peak (dose*=mask leaves every in-body voxel
untouched) — physically it is a clip, not a model. The photon pipeline instead uses
`raytrace_skinentry`: it references depth from where each ray first enters the body. This
file does the proton analogue for the PB prior:

  * WEPL is accumulated ONLY after the ray (source->voxel) first crosses density>skin_thr
    (the skin), so the depth origin is the per-ray skin entry (not the single central-ray
    SSD, and not the source). For oblique / curved-surface geometry the peak position can
    therefore shift — which the mask cannot do.
  * Voxels still upstream of the skin along their own ray get `entered=0` and hence zero
    dose (no IDD(0) painted in the air gap) — by construction, not by a whole-body mask.
  * The `rad_depth_offset` air-correction is dropped, since the depth is now measured
    directly from the skin.

New file; does NOT modify committed proton/CT code. Reuses ProtonMachineData / _interp_1d /
_compute_ssd from proton_pb_gpu.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from doserad.physics.density import hu_to_density
from doserad.physics.proton_pb_gpu import (ProtonMachineData, _interp_1d, _compute_ssd,
                                           _SSD_DENSITY_THRESHOLD)

_SKIN_THR = 0.05   # same skin threshold as the SSD detection / photon skin-entry


def _wepl_crop_skinentry(density, spacing, origin, src, coords, dev, skin_thr=_SKIN_THR,
                         march_start_mm: float = 0.0, step_mm: float = 1.0, chunk: int = 8_000_000):
    """Per-ray skin-referenced WEPL (g/cm^2) + `entered` flag for every voxel in `coords`.

    March source->voxel; accumulate density*step only AFTER the cumulative density along the
    ray first exceeds `skin_thr`. `entered` = the ray crossed the skin at or before the voxel.
    `march_start_mm` must sit just upstream of the skin (we pass central-ray SSD - 100mm).
    """
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens5 = (density.to(dev, torch.float32) if torch.is_tensor(density)
             else torch.as_tensor(density, dtype=torch.float32, device=dev)).view(1, 1, nz, ny, nx)
    src_t = torch.as_tensor(src, dtype=torch.float32, device=dev)

    shp = coords.shape[:-1]
    P = coords.reshape(-1, 3)
    vec = P - src_t
    dist = torch.linalg.norm(vec, dim=-1)
    direction = vec / dist.clamp_min(1e-6).unsqueeze(-1)

    n_steps = int(torch.ceil((dist.max() - march_start_mm) / step_mm).item()) + 1
    if n_steps < 1:
        march_start_mm = max(float(dist.min().item()) - 50.0, 0.0)
        n_steps = max(int(torch.ceil((dist.max() - march_start_mm) / step_mm).item()) + 1, 1)

    inv_w = 2.0 / max(sx * (nx - 1), 1e-6)
    inv_h = 2.0 / max(sy * (ny - 1), 1e-6)
    inv_d = 2.0 / max(sz * (nz - 1), 1e-6)
    t = march_start_mm + (torch.arange(n_steps, device=dev, dtype=torch.float32) + 0.5) * step_mm
    out = torch.zeros(P.shape[0], dtype=torch.float32, device=dev)
    entered = torch.zeros(P.shape[0], dtype=torch.float32, device=dev)
    csz = max(1, chunk // max(n_steps, 1))
    for s0 in range(0, P.shape[0], csz):
        s1 = min(s0 + csz, P.shape[0])
        d_c = direction[s0:s1]; dist_c = dist[s0:s1]; m = d_c.shape[0]
        pts = src_t.view(1, 1, 3) + t.view(1, -1, 1) * d_c.view(m, 1, 3)
        gx = (pts[..., 0] - ox) * inv_w - 1
        gy = (pts[..., 1] - oy) * inv_h - 1
        gz = (pts[..., 2] - oz) * inv_d - 1
        grid = torch.stack([gx, gy, gz], dim=-1).view(1, 1, m, n_steps, 3)
        sampled = torch.nn.functional.grid_sample(
            dens5, grid, mode="bilinear", align_corners=True, padding_mode="zeros").view(m, n_steps)
        active = (t.view(1, -1) < dist_c.view(-1, 1)).float()           # samples up to the voxel
        hit = (sampled > skin_thr).float() * active                    # skin hits before the voxel
        crossed = (torch.cumsum(hit, dim=1) > 0).float()               # 1 from first skin hit onward
        out[s0:s1] = (sampled * crossed * active).sum(dim=1) * (step_mm / 10.0)   # WEPL from skin
        entered[s0:s1] = (hit.sum(dim=1) > 0).float()                  # ray reached skin by the voxel
    return out.reshape(shp), entered.reshape(shp)


def proton_pb_dose_gpu_skinentry(image, ray_source, ray_target, energy, out_bbox,
                                 machine: ProtonMachineData, hu_anchors=None, density_override=None,
                                 comp_fac: float = 1.0, device: str | None = None,
                                 skin_thr: float = _SKIN_THR, return_tensor: bool = False):
    """Proton PB dose on crop `out_bbox`=(z0,z1,y0,y1,x0,x1) with per-ray skin-entry depth."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    if density_override is not None:
        density = (density_override.to(dev, torch.float32) if torch.is_tensor(density_override)
                   else np.asarray(density_override, dtype=np.float32))
    else:
        density = hu_to_density(image.array, hu_anchors).astype(np.float32)

    tgt = np.asarray(ray_target, dtype=np.float64)
    jsrc = np.asarray(ray_source, dtype=np.float64)
    axis = tgt - jsrc
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * machine.sad).astype(np.float32)   # machine-SAD source (matches pyRadPlan)
    axis_f = axis.astype(np.float32)

    z0, z1, y0, y1, x0, x1 = out_bbox
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)

    ssd = _compute_ssd(density.detach() if torch.is_tensor(density) else density,
                       image.spacing, image.origin, src, axis_f, machine.sad, dev,
                       threshold=_SSD_DENSITY_THRESHOLD)

    # per-ray skin-referenced WEPL + entered flag; start the march just upstream of the skin
    wepl, entered = _wepl_crop_skinentry(density, image.spacing, image.origin, src, coords, dev,
                                         skin_thr=skin_thr, march_start_mm=max(ssd - 100.0, 0.0))

    src_t = torch.as_tensor(src, device=dev); axis_t = torch.as_tensor(axis_f, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)

    eidx = machine.energy_index(energy)
    n = int(machine.lengths[eidx].item())
    depths = machine.depths[eidx, :n]
    idd = machine.conversion_factor * machine.idd[eidx, :n]
    sigma_d = machine.sigma[eidx, :n]
    offset = float(machine.offset[eidx].item())
    sigma_ini_sq = machine.sigma_ini(eidx, ssd) ** 2

    rad_depths_mm = wepl * 10.0                          # g/cm^2 -> mm water, referenced from SKIN
    eff_depths = depths + offset                         # NO rad_depth_offset (skin-referenced)
    d = rad_depths_mm.reshape(-1)

    idd_v = _interp_1d(d, eff_depths, idd)
    sig_v = _interp_1d(d, eff_depths, sigma_d)
    out_of_range = (d > eff_depths[-1]) | (d < eff_depths[0])
    idd_v = torch.where(out_of_range, torch.zeros_like(idd_v), idd_v)

    r2 = (lateral_dist.reshape(-1)) ** 2
    if os.environ.get("DOSERAD_LATERAL_DOUBLE") == "1" and machine.sigma1 is not None:
        # Hong DOUBLE-Gaussian lateral (nuclear halo); mirrors accel/proton_build_ray. OPT-IN; default single.
        sig1_v = _interp_1d(d, eff_depths, machine.sigma1[eidx, :n])
        sig2_v = _interp_1d(d, eff_depths, machine.sigma2[eidx, :n])
        w_v = _interp_1d(d, eff_depths, machine.weight[eidx, :n]).clamp(0.0, 1.0)
        s1sq = sig1_v ** 2 + sigma_ini_sq; s2sq = sig2_v ** 2 + sigma_ini_sq
        g1 = torch.exp(-r2 / (2.0 * s1sq)) / (2.0 * torch.pi * s1sq)
        g2 = torch.exp(-r2 / (2.0 * s2sq)) / (2.0 * torch.pi * s2sq)
        lateral = (1.0 - w_v) * g1 + w_v * g2
    else:
        sigma_sq = sig_v ** 2 + sigma_ini_sq
        lateral = torch.exp(-r2 / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
    dose = comp_fac * lateral * idd_v
    dose = dose.reshape(rad_depths_mm.shape).clamp_min(0.0)
    dose = dose * entered                                # per-ray: zero voxels not yet past the skin

    if return_tensor:
        return dose
    return dose.cpu().numpy().astype(np.float32)
