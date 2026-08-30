"""GPU-resident photon channel builder — a bit-identical, faster drop-in for
doserad.physics.channels.photon_channels(return_tensor=True) for the DEPLOY inner loop.

photon_channels re-runs `density_override.astype(float32)` (a full ~60 MB CPU copy, ~30 ms) AND
`torch.as_tensor(density, device=cuda)` (~6 ms H2D) EVERY control point, even though density is
constant per patient — ~36 ms/CP of pure waste (measured). radiological_depth_fast re-uploads the
density too (~6 ms). This builder takes the density already on the GPU (uploaded ONCE per patient)
and reuses it, saving ~37 ms/CP. The channel MATH is copied verbatim from channels.py, so the output
is bit-identical (verified) — this is a lossless memory-traffic optimization, not a physics change.
"""
from __future__ import annotations

import numpy as np
import torch

from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.channels import _mask_bbox, MU_6MV_PER_CM
from doserad.physics.raytrace import radiological_depth_fast


@torch.no_grad()
def photon_channels_fast(image, machine, iso_xyz, gantry_deg, mlc_left, mlc_right, *,
                         dens_t: torch.Tensor, coords: torch.Tensor, crop_margin: int = 8,
                         dev: str = "cuda"):
    """dens_t: density (z,y,x) float32 GPU tensor (uploaded once). coords: (z,y,x,3) world GPU tensor
    (_build_coords, once per image). Returns (stack[5,d,h,w] GPU, bbox) == photon_channels(return_tensor)."""
    nz, ny, nx = image.array.shape
    iso = np.asarray(iso_xyz, dtype=np.float64)
    src_np = beam_source_pos(iso, machine.sad_mm, gantry_deg)
    axis_np, u_hat_np, v_hat_np = beam_basis(gantry_deg)

    src = torch.as_tensor(src_np, dtype=torch.float32, device=dev)
    axis = torch.as_tensor(axis_np, dtype=torch.float32, device=dev)
    u_hat = torch.as_tensor(u_hat_np, dtype=torch.float32, device=dev)
    v_hat = torch.as_tensor(v_hat_np, dtype=torch.float32, device=dev)
    iso_t = torch.as_tensor(iso, dtype=torch.float32, device=dev)

    coords_full = coords
    vec = coords_full - src
    denom = (vec * axis).sum(-1)
    t_iso = ((iso_t - src) * axis).sum() / torch.where(
        denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    hit = src + t_iso.unsqueeze(-1) * vec
    rel = hit - iso_t
    u = (rel * u_hat).sum(-1)
    v = (rel * v_hat).sum(-1)

    half = machine.num_leaf_pairs / 2.0
    pair = torch.floor(v / machine.leaf_thickness_mm + half).long()
    valid = (pair >= 0) & (pair < machine.num_leaf_pairs)
    pidx = pair.clamp(0, machine.num_leaf_pairs - 1)
    ml_t = torch.as_tensor(mlc_left, dtype=torch.float32, device=dev)
    mr_t = torch.as_tensor(mlc_right, dtype=torch.float32, device=dev)
    left_t = ml_t[pidx]; right_t = mr_t[pidx]
    jx, jy = machine.jaw_x_mm, machine.jaw_y_mm
    open_mask = (valid & (left_t < right_t) &
                 (u >= left_t) & (u <= right_t) &
                 (u >= jx[0]) & (u <= jx[1]) &
                 (v >= jy[0]) & (v <= jy[1])).float()

    bbox = _mask_bbox(open_mask, crop_margin, (nz, ny, nx))
    z0, z1, y0, y1, x0, x1 = bbox
    sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    open_mask = open_mask[sl]
    dens_c = dens_t[sl]
    coords_c = coords_full[(*sl, slice(None))]

    # rdepth: pass the GPU density (radiological_depth_fast accepts a tensor -> no re-upload).
    rdepth = radiological_depth_fast(dens_t, image.spacing, image.origin, src_np,
                                     axis_np, u_hat_np, v_hat_np, iso, out_bbox=bbox, coords=coords_full)
    rdepth_t = torch.as_tensor(rdepth, dtype=torch.float32, device=dev)

    fluence = open_mask * torch.exp(-MU_6MV_PER_CM * rdepth_t)
    rel_iso = coords_c - iso_t
    perp = rel_iso - ((rel_iso * axis).sum(-1).unsqueeze(-1) * axis)
    dist_to_cax = torch.linalg.norm(perp, dim=-1)
    source_dist = torch.linalg.norm(coords_c - src, dim=-1)

    stack = torch.stack([dens_c, rdepth_t, fluence, dist_to_cax, source_dist], dim=0)
    return stack, bbox
