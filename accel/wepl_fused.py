"""Fused WEPL raymarch: one grid_sample yields BOTH the source-referenced WEPL (ch1 channel)
and the skin-referenced WEPL + entered flag (PB engine), which the deploy path computes as two
separate raymarches. The march grid + density sampling (the expensive kernel) is shared.

Byte-equivalent to _wepl_crop (source) and _wepl_crop_skinentry (skin) when marched from the
earlier start; the extra upstream air (rho~0) contributes ~0 to the source integral.
"""
from __future__ import annotations

import torch


def wepl_fused(density, spacing, origin, src, coords, dev, skin_thr=0.05,
               march_start_mm: float = 0.0, step_mm: float = 1.0, chunk: int = 8_000_000):
    """Returns (wepl_source, wepl_skin, entered), each shaped like coords[...,0]."""
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
    w_src = torch.zeros(P.shape[0], dtype=torch.float32, device=dev)
    w_skin = torch.zeros(P.shape[0], dtype=torch.float32, device=dev)
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
        active = (t.view(1, -1) < dist_c.view(-1, 1)).float()
        sa = sampled * active
        w_src[s0:s1] = sa.sum(dim=1) * (step_mm / 10.0)                 # source-referenced (ch1)
        hit = (sampled > skin_thr).float() * active
        crossed = (torch.cumsum(hit, dim=1) > 0).float()
        w_skin[s0:s1] = (sa * crossed).sum(dim=1) * (step_mm / 10.0)    # skin-referenced (PB)
        entered[s0:s1] = (hit.sum(dim=1) > 0).float()
    return w_src.reshape(shp), w_skin.reshape(shp), entered.reshape(shp)
