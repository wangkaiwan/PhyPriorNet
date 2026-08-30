"""Differentiable photon physics channels from a predicted (synthetic) CT — for the end-to-end
dose-aware MRI->sCT->dose model. A dose loss can back-propagate THROUGH these channels into the
synthesis network, so the sCT is optimized to be dose-correct (not just image-similar).

Everything here is torch-autograd compatible and matches the precompute semantics
(doserad/physics/channels.py + raytrace.py) so v13 dose-net weights transfer:
  - hu_to_density_torch : differentiable piecewise-linear HU->density (matches np.interp/hu_to_density)
  - radiological_depth_fast_torch : the BEV-fan ray-trace (grid_sample + cumsum) returning a TENSOR
    (the geometry-extent scalars u_max/v_max/d_max are constants w.r.t. density -> .item() is fine;
     gradient flows through the sampled density values)
  - photon_density_channels : assemble [density, rdepth, fluence(, naive)] on the bbox crop,
    differentiable w.r.t. the input density (= w.r.t. the predicted sCT). Geometry channels
    (dist_to_cax, source_dist) are density-independent and supplied separately (cache / on-the-fly).

NOTE: NEW file. Does not modify the committed precompute/inference channel code.
"""
from __future__ import annotations
import numpy as np
import torch

# naive-prior constants (must match doserad/physics/naive_dose.py)
SAD_MM = 1000.0
_SURFACE = 0.2
_TAU = 0.5
# fluence attenuation coefficient (6 MV) — MUST match channels.py MU_6MV_PER_CM
MU_6MV = 0.0475   # applied to rdepth (g/cm^2 ~ cm water); fluence = open_mask * exp(-MU_6MV * rdepth)


def fluence_torch(open_mask: torch.Tensor, rdepth: torch.Tensor) -> torch.Tensor:
    """Differentiable primary fluence (matches channels.py). open_mask is geometry (constant)."""
    return open_mask * torch.exp(-MU_6MV * rdepth)


def hu_to_density_torch(hu: torch.Tensor, anchors) -> torch.Tensor:
    """Differentiable piecewise-linear HU->density. `anchors` = sorted ((hu,rho),...).
    Matches np.interp (clamps outside the anchor range). Grad flows w.r.t. `hu`."""
    dev, dt = hu.device, hu.dtype
    xp = torch.tensor([a[0] for a in anchors], device=dev, dtype=dt)
    fp = torch.tensor([a[1] for a in anchors], device=dev, dtype=dt)
    huc = hu.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, huc, right=True).clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]; x1 = xp[idx]; y0 = fp[idx - 1]; y1 = fp[idx]
    w = (huc - x0) / (x1 - x0).clamp_min(1e-6)
    return y0 + w * (y1 - y0)


def radiological_depth_fast_torch(dens: torch.Tensor, spacing, origin, source_xyz, axis,
                                  u_hat, v_hat, iso_xyz, n_u=128, n_v=128, n_d=256,
                                  out_bbox=None, coords: torch.Tensor | None = None) -> torch.Tensor:
    """Autograd port of raytrace.radiological_depth_fast — same math, returns a tensor.
    `dens` is a (nz,ny,nx) torch tensor (may require grad). Geometry args are constants."""
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

    u_max = float(vu.abs().max()) * 1.05 + 1.0       # geometry constants (no grad needed)
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
        align_corners=True, padding_mode="border").view(n_v, n_u, n_d)
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


def naive_dose_torch(density, rdepth, fluence, source_dist, skin_gate: bool = False) -> torch.Tensor:
    """Differentiable naive prior (matches naive_dose.compute_naive_dose, scatter=False).

    `skin_gate` (default False = OFF): mirror of the numpy compute_naive_dose entered-gate.
    Zeros the prior where rdepth==0 (external air upstream of the skin); never masks in-body
    voxels. MUST be paired with a SKIN-ENTRY rdepth (radiological_depth_skinentry_torch), where
    rdepth==0 iff the ray has not yet crossed the skin (identical semantics to the CT path)."""
    inv_sq = (SAD_MM / source_dist.clamp_min(1.0)) ** 2
    buildup = 1.0 - (1.0 - _SURFACE) * torch.exp(-rdepth / _TAU)
    out = fluence * inv_sq * buildup
    if skin_gate:
        out = out * (rdepth > 0).to(out.dtype)
    return out
