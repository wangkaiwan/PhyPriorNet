"""Proton build acceleration — step 1: per-RAY geometry sharing (exact, ~2x on WEPL+PB).

A ray's beamlets (same ray_source/ray_target, different energy) share ALL geometry: src, SSD,
WEPL, lateral distance, sigma_ini, rad_depth_offset (SSD-only). Only the energy kernel lookup
(idd/sigma tables) and the final gaussian*idd differ. The deployed path recomputes the whole
geometry per beamlet; here we compute it ONCE on the ray's union bbox and produce each beamlet's
dose by slicing + a per-energy kernel eval. Byte-equivalent to the per-beamlet engine.

pb_dose_ray(image, ray_source, ray_target, beamlets, machine, density_override) ->
    dict {output_idx or list-index -> dose_crop_np (dz,dy,dx)} in the same order as `beamlets`,
    where each beamlet = dict(energy=float, bbox=(z0,z1,y0,y1,x0,x1)).
"""
from __future__ import annotations

import numpy as np
import torch

from doserad.physics.proton_pb_gpu import (ProtonMachineData, _wepl_crop, _compute_ssd,
                                           _interp_1d, _SSD_DENSITY_THRESHOLD)
from doserad.physics.density import hu_to_density


def _union_bbox(bboxes):
    z0 = min(b[0] for b in bboxes); z1 = max(b[1] for b in bboxes)
    y0 = min(b[2] for b in bboxes); y1 = max(b[3] for b in bboxes)
    x0 = min(b[4] for b in bboxes); x1 = max(b[5] for b in bboxes)
    return (z0, z1, y0, y1, x0, x1)


def pb_dose_ray(image, ray_source, ray_target, beamlets, *, machine: ProtonMachineData,
                hu_anchors=None, density_override=None, comp_fac: float = 1.0,
                device: str | None = None, return_tensor: bool = False):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin
    nz, ny, nx = image.array.shape

    if density_override is not None:
        density = (density_override.to(dev, torch.float32) if torch.is_tensor(density_override)
                   else np.asarray(density_override, dtype=np.float32))
    else:
        density = hu_to_density(image.array, hu_anchors).astype(np.float32)
    dens_is_tensor = torch.is_tensor(density)

    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = (tgt - jsrc); axis = axis / (np.linalg.norm(axis) + 1e-12)
    src = (tgt - axis * machine.sad).astype(np.float32)          # use_machine_sad_source=True
    axis_f = axis.astype(np.float32)

    # ---- geometry ONCE on the ray's union bbox ----
    ubb = _union_bbox([b["bbox"] for b in beamlets])
    z0, z1, y0, y1, x0, x1 = ubb
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)                   # (Dz,Dy,Dx,3)

    ssd = _compute_ssd(density.detach() if dens_is_tensor else density, image.spacing,
                       image.origin, src, axis_f, machine.sad, dev, threshold=_SSD_DENSITY_THRESHOLD)
    wepl = _wepl_crop(density, image.spacing, image.origin, src, coords, dev,
                      march_start_mm=max(ssd - 50.0, 0.0))       # (Dz,Dy,Dx)
    src_t = torch.as_tensor(src, device=dev); axis_t = torch.as_tensor(axis_f, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)        # (Dz,Dy,Dx)

    sigma_ini = machine.sigma_ini_from_ssd(ssd) if hasattr(machine, "sigma_ini_from_ssd") else None
    rad_depths_mm = wepl * 10.0
    r2_full = lateral_dist ** 2

    out = {}
    for i, bl in enumerate(beamlets):
        eidx = machine.energy_index(bl["energy"])
        n = int(machine.lengths[eidx].item())
        depths = machine.depths[eidx, :n]; idd = machine.conversion_factor * machine.idd[eidx, :n]
        sigma_d = machine.sigma[eidx, :n]; offset = float(machine.offset[eidx].item())
        sig_ini = machine.sigma_ini(eidx, ssd); sig_ini_sq = sig_ini ** 2
        rad_depth_offset = 0.0011 * (ssd + machine.bams_to_iso_dist - machine.sad - machine.fit_air_offset)
        eff_depths = depths + offset - rad_depth_offset
        # slice this beamlet's own bbox out of the union
        bz0, bz1, by0, by1, bx0, bx1 = bl["bbox"]
        sl = (slice(bz0 - z0, bz1 - z0 + 1), slice(by0 - y0, by1 - y0 + 1), slice(bx0 - x0, bx1 - x0 + 1))
        d = rad_depths_mm[sl].reshape(-1)
        idd_v = _interp_1d(d, eff_depths, idd); sig_v = _interp_1d(d, eff_depths, sigma_d)
        oor = (d > eff_depths[-1]) | (d < eff_depths[0])
        idd_v = torch.where(oor, torch.zeros_like(idd_v), idd_v)
        sigma_sq = sig_v ** 2 + sig_ini_sq
        r2 = r2_full[sl].reshape(-1)
        lateral = torch.exp(-r2 / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
        dose = (comp_fac * lateral * idd_v).reshape(rad_depths_mm[sl].shape).clamp_min(0.0)
        out[i] = dose if return_tensor else dose.cpu().numpy().astype(np.float32)
    return out
