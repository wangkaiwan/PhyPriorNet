"""Differentiable (autograd, torch) SKIN-ENTRY radiological depth for the MRI synthesis path.

The MRI synthesis training recomputes the photon physics on-the-fly from the *synthesized* density
(diff_channels.radiological_depth_fast_torch). To stay consistent with the skin-entry photon dose
engine (ft_skinentry_photonct_*, trained on channels_skinentry / raytrace_skinentry), the on-the-fly
torch ray-trace must ALSO reference depth from the skin: density is accumulated only after the ray
first crosses density > skin_thr, and the fan is sampled with padding_mode='zeros' (external air = 0).
This is the exact torch analogue of raytrace_skinentry (numpy). New file; does not modify the
committed diff_channels / CT-photon code. Other torch channel helpers are re-exported unchanged.
"""
from __future__ import annotations

import torch

from doserad.physics.diff_channels import (fluence_torch, hu_to_density_torch,  # noqa: F401
                                           naive_dose_torch)

_SKIN_THR = 0.05   # same skin threshold as raytrace_skinentry / SSD detection


def radiological_depth_skinentry_torch(dens: torch.Tensor, spacing, origin, source_xyz, axis,
                                       u_hat, v_hat, iso_xyz, n_u=128, n_v=128, n_d=256,
                                       out_bbox=None, coords: torch.Tensor | None = None,
                                       skin_thr: float = _SKIN_THR) -> torch.Tensor:
    """Autograd skin-entry radiological depth: depth origin is the per-ray skin entry, not the
    source. Same BEV-fan math as diff_channels.radiological_depth_fast_torch, with two changes:
    (1) fan density sampled with padding_mode='zeros'; (2) before the along-ray cumsum, density
    upstream of the first skin crossing is zeroed (hard cumulative gate, as in raytrace_skinentry)."""
    dev, dt = dens.device, dens.dtype
    nz, ny, nx = dens.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    tens = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
    src = tens(source_xyz); axis = tens(axis); u_hat = tens(u_hat); v_hat = tens(v_hat); iso = tens(iso_xyz)

    if coords is None:
        xs = ox + torch.arange(nx, device=dev, dtype=dt) * sx
        ys = oy + torch.arange(ny, device=dev, dtype=dt) * sy
        zs = oz + torch.arange(nz, device=dev, dtype=dt) * sz
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        P = torch.stack([gx, gy, gz], dim=-1)
    else:
        P = coords
    vec = P - src
    denom = (vec * axis).sum(-1)
    t = ((iso - src) * axis).sum() / torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    hit = src + t.unsqueeze(-1) * vec
    rel = hit - iso
    vu = (rel * u_hat).sum(-1); vv = (rel * v_hat).sum(-1)
    vdist = torch.linalg.norm(vec, dim=-1)

    u_max = float(vu.abs().max()) * 1.05 + 1.0
    v_max = float(vv.abs().max()) * 1.05 + 1.0
    d_min = 0.0; d_max = float(vdist.max()) * 1.02
    us = torch.linspace(-u_max, u_max, n_u, device=dev, dtype=dt)
    vs = torch.linspace(-v_max, v_max, n_v, device=dev, dtype=dt)
    ds = torch.linspace(d_min, d_max, n_d, device=dev, dtype=dt)
    step_cm = (d_max - d_min) / (n_d - 1) / 10.0

    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    plane_pt = iso.view(1, 1, 3) + gu.unsqueeze(-1) * u_hat.view(1, 1, 3) + gv.unsqueeze(-1) * v_hat.view(1, 1, 3)
    ray_dir = plane_pt - src.view(1, 1, 3)
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    bev_pts = src.view(1, 1, 1, 3) + ds.view(1, 1, n_d, 1) * ray_dir.unsqueeze(2)

    def w2n(p):
        gx_ = (p[..., 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1
        gy_ = (p[..., 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1
        gz_ = (p[..., 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1
        return torch.stack([gx_, gy_, gz_], dim=-1)

    dens5 = dens.view(1, 1, nz, ny, nx)
    bev_dens = torch.nn.functional.grid_sample(
        dens5, w2n(bev_pts).view(1, n_v, n_u, n_d, 3), mode="bilinear",
        align_corners=True, padding_mode="zeros").view(n_v, n_u, n_d)   # (skin-entry) air outside = 0
    # --- SKIN-ENTRY gate: zero density upstream of the first skin crossing along each ray ---
    entered = (torch.cumsum((bev_dens > skin_thr).to(dt), dim=-1) > 0).to(dt)
    bev_dens = bev_dens * entered
    bev_rdepth = (torch.cumsum(bev_dens, dim=-1) - 0.5 * bev_dens) * step_cm

    if out_bbox is not None:
        z0, z1, y0, y1, x0, x1 = out_bbox
        vu = vu[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vv = vv[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vdist = vdist[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    oz_, oy_, ox_ = vu.shape
    q = torch.stack([(vdist - d_min) / max(d_max - d_min, 1e-6) * 2 - 1, vu / u_max, vv / v_max], dim=-1)
    out = torch.nn.functional.grid_sample(
        bev_rdepth.view(1, 1, n_v, n_u, n_d), q.view(1, oz_, oy_, ox_, 3), mode="bilinear",
        align_corners=True, padding_mode="border").view(oz_, oy_, ox_)
    return out
