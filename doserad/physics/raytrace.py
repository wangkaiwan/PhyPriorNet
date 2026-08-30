"""Torch ray-marched radiological depth: line integral of density from the
source to each voxel center, returned in g/cm^2 (density g/cm^3 * cm).

Vectorized over all voxels: march N fixed steps from the source toward each
voxel, trilinearly sampling density at each step and accumulating. Runs on
GPU when available; returns a numpy array (z,y,x)."""
from __future__ import annotations

import numpy as np
import torch


def radiological_depth(density: np.ndarray, spacing, origin,
                       source_xyz, step_mm: float = 1.0,
                       device: str | None = None) -> np.ndarray:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    dens = torch.as_tensor(density, dtype=torch.float32, device=dev)

    xs = ox + torch.arange(nx, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(ny, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(nz, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    target = torch.stack([gx, gy, gz], dim=-1)            # (z,y,x,3) world xyz
    src = torch.as_tensor(source_xyz, dtype=torch.float32, device=dev)

    vec = target - src
    dist_mm = torch.linalg.norm(vec, dim=-1)
    n_steps = int(torch.ceil(dist_mm.max() / step_mm).item())
    direction = vec / dist_mm.clamp_min(1e-6).unsqueeze(-1)

    acc = torch.zeros((nz, ny, nx), dtype=torch.float32, device=dev)

    def to_norm(p):
        # world xyz -> grid_sample normalized coords
        # grid_sample with input (1,1,nz,ny,nx) expects grid (...,3) = (x,y,z)
        # where x indexes W=nx, y indexes H=ny, z indexes D=nz
        # normalized: -1 -> index 0, +1 -> index (dim-1)
        gx_ = (p[..., 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1  # x -> W=nx
        gy_ = (p[..., 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1  # y -> H=ny
        gz_ = (p[..., 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1  # z -> D=nz
        return torch.stack([gx_, gy_, gz_], dim=-1)

    # grid_sample input shape: (N,C,D,H,W) = (1,1,nz,ny,nx)
    dens5 = dens.view(1, 1, nz, ny, nx)
    for k in range(n_steps):
        t = (k + 0.5) * step_mm
        active = (t < dist_mm).float()
        pts = src + direction * t
        norm = to_norm(pts).view(1, nz, ny, nx, 3)
        sampled = torch.nn.functional.grid_sample(
            dens5, norm, mode="bilinear", align_corners=True,
            padding_mode="border").view(nz, ny, nx)
        acc = acc + sampled * active * (step_mm / 10.0)   # mm -> cm
    return acc.cpu().numpy()


def radiological_depth_fast(density: np.ndarray, spacing, origin,
                            source_xyz, axis, u_hat, v_hat, iso_xyz,
                            n_u: int = 128, n_v: int = 128, n_d: int = 256,
                            device: str | None = None,
                            out_bbox: tuple | None = None,
                            coords: "torch.Tensor | None" = None) -> np.ndarray:
    """O(volume) radiological depth via divergent beam-eye-view cumsum.

    Resamples density onto a (n_v, n_u, n_d) BEV grid of rays from `source`
    through iso-plane points iso + u*u_hat + v*v_hat, cumulative-sums along
    depth (g/cm^2), then samples back to every patient voxel via its own
    iso-plane (u,v) and along-ray depth. Matches `radiological_depth` within
    interpolation tolerance.

    `out_bbox` = (z0,z1,y0,y1,x0,x1) inclusive: when given, the BEV fan is still
    derived from the FULL volume (so values are bit-identical to the full call),
    but the final per-voxel sampling is restricted to the bbox subgrid — this is
    the runtime fast-path (the network only needs the beam bbox). Returns the
    bbox-shaped (dz,dy,dx) array."""
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
        P = coords                                  # per-image reuse (z,y,x,3)
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
    d_min = 0.0  # integrate from source (d=0) so cumsum at voxel depth = total radiological depth
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
    bev_pts = (src.view(1, 1, 1, 3)
               + ds.view(1, 1, n_d, 1) * ray_dir.unsqueeze(2))

    def world_to_norm(p):
        gx_ = (p[..., 0] - ox) / max(sx * (nx - 1), 1e-6) * 2 - 1
        gy_ = (p[..., 1] - oy) / max(sy * (ny - 1), 1e-6) * 2 - 1
        gz_ = (p[..., 2] - oz) / max(sz * (nz - 1), 1e-6) * 2 - 1
        return torch.stack([gx_, gy_, gz_], dim=-1)

    dens5 = dens.view(1, 1, nz, ny, nx)
    bev_norm = world_to_norm(bev_pts).view(1, n_v, n_u, n_d, 3)
    bev_dens = torch.nn.functional.grid_sample(
        dens5, bev_norm, mode="bilinear", align_corners=True,
        padding_mode="border").view(n_v, n_u, n_d)
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
