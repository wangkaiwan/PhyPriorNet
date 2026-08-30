"""Skin-entry radiological depth (EXPERIMENT, new file — does NOT touch the
existing CT-dose pipeline in raytrace.py).

Two physically-motivated changes vs `radiological_depth_fast`:
  1. `padding_mode="zeros"` for the density BEV sample: outside the CT FOV the
     ray sees vacuum (0), not a replicated border voxel — removes the
     border-padding artifact (the dominant "extreme case" error, up to ~4-11 g/cm^2).
  2. Per-ray SKIN ENTRY: along each ray, density is zeroed until the first voxel
     above `skin_thr` (g/cm^3). External air (rho~0.0012) before the patient
     therefore contributes nothing; all internal tissue INCLUDING lung is kept
     (lung rho~0.2-0.4 >> skin_thr), so no body-mask / no tissue is cut.
     (There is no couch in this dataset; a real couch, being physical, would be
     kept by NOT using this on couch beams — not relevant here.)

Same divergent BEV-cumsum geometry as raytrace.radiological_depth_fast, so values
are identical except for the two changes above."""
from __future__ import annotations

import numpy as np
import torch


def radiological_depth_skinentry(density: np.ndarray, spacing, origin,
                                 source_xyz, axis, u_hat, v_hat, iso_xyz,
                                 n_u: int = 128, n_v: int = 128, n_d: int = 256,
                                 skin_thr: float = 0.05,
                                 device: str | None = None,
                                 out_bbox: tuple | None = None,
                                 coords: "torch.Tensor | None" = None) -> np.ndarray:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens = torch.as_tensor(density, dtype=torch.float32, device=dev)
    src = torch.as_tensor(source_xyz, dtype=torch.float32, device=dev)
    axis = torch.as_tensor(axis, dtype=torch.float32, device=dev)
    u_hat = torch.as_tensor(u_hat, dtype=torch.float32, device=dev)
    v_hat = torch.as_tensor(v_hat, dtype=torch.float32, device=dev)
    iso = torch.as_tensor(iso_xyz, dtype=torch.float32, device=dev)

    if coords is None:
        xs = ox + torch.arange(nx, device=dev, dtype=torch.float32) * sx
        ys = oy + torch.arange(ny, device=dev, dtype=torch.float32) * sy
        zs = oz + torch.arange(nz, device=dev, dtype=torch.float32) * sz
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        P = torch.stack([gx, gy, gz], dim=-1)
    else:
        P = coords
    vec = P - src
    denom = (vec * axis).sum(-1)
    t = ((iso - src) * axis).sum() / torch.where(denom.abs() < 1e-6,
                                                 torch.full_like(denom, 1e-6), denom)
    hit = src + t.unsqueeze(-1) * vec
    rel = hit - iso
    vu = (rel * u_hat).sum(-1)
    vv = (rel * v_hat).sum(-1)
    vdist = torch.linalg.norm(vec, dim=-1)

    u_max = float(vu.abs().max()) * 1.05 + 1.0
    v_max = float(vv.abs().max()) * 1.05 + 1.0
    d_min = 0.0
    d_max = float(vdist.max()) * 1.02
    us = torch.linspace(-u_max, u_max, n_u, device=dev)
    vs = torch.linspace(-v_max, v_max, n_v, device=dev)
    ds = torch.linspace(d_min, d_max, n_d, device=dev)
    step_cm = (d_max - d_min) / (n_d - 1) / 10.0

    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    plane_pt = (iso.view(1, 1, 3)
                + gu.unsqueeze(-1) * u_hat.view(1, 1, 3)
                + gv.unsqueeze(-1) * v_hat.view(1, 1, 3))
    ray_dir = plane_pt - src.view(1, 1, 3)
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    bev_pts = (src.view(1, 1, 1, 3) + ds.view(1, 1, n_d, 1) * ray_dir.unsqueeze(2))

    def world_to_norm(p):
        gx_ = (p[..., 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1
        gy_ = (p[..., 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1
        gz_ = (p[..., 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1
        return torch.stack([gx_, gy_, gz_], dim=-1)

    dens5 = dens.view(1, 1, nz, ny, nx)
    bev_norm = world_to_norm(bev_pts).view(1, n_v, n_u, n_d, 3)
    # CHANGE 1: zeros padding -> outside FOV = vacuum, no border replication
    bev_dens = torch.nn.functional.grid_sample(
        dens5, bev_norm, mode="bilinear", align_corners=True,
        padding_mode="zeros").view(n_v, n_u, n_d)
    # CHANGE 2: per-ray skin entry -> zero density before the first >skin_thr crossing
    entered = (torch.cumsum((bev_dens > skin_thr).float(), dim=-1) > 0).float()
    bev_dens = bev_dens * entered
    bev_rdepth = (torch.cumsum(bev_dens, dim=-1) - 0.5 * bev_dens) * step_cm

    if out_bbox is not None:
        z0, z1, y0, y1, x0, x1 = out_bbox
        vu = vu[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vv = vv[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        vdist = vdist[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    oz_, oy_, ox_ = vu.shape
    qn_u = vu / u_max
    qn_v = vv / v_max
    qn_d = (vdist - d_min) / max(d_max - d_min, 1e-6) * 2 - 1
    q = torch.stack([qn_d, qn_u, qn_v], dim=-1).view(1, oz_, oy_, ox_, 3)
    out = torch.nn.functional.grid_sample(
        bev_rdepth.view(1, 1, n_v, n_u, n_d), q, mode="bilinear",
        align_corners=True, padding_mode="border").view(oz_, oy_, ox_)
    return out.cpu().numpy()
