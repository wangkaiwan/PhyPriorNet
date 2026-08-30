"""Photon channel stack using the SKIN-ENTRY radiological depth (experiment).
Identical to doserad.physics.channels.photon_channels except rdepth (and hence
fluence = open_mask * exp(-mu*rdepth)) use radiological_depth_skinentry.
New file — does not touch the existing pipeline."""
from __future__ import annotations

import numpy as np
import torch

from doserad.io.mha import Volume
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import PhotonMachine
from doserad.physics.channels import MU_6MV_PER_CM, _mask_bbox
from doserad.physics.raytrace_skinentry import radiological_depth_skinentry


def photon_channels_skinentry(image: Volume, machine: PhotonMachine,
                              iso_xyz, gantry_deg: float,
                              mlc_left: np.ndarray, mlc_right: np.ndarray,
                              *, density_override: np.ndarray | None = None,
                              crop_margin: int | None = None,
                              coords: "torch.Tensor | None" = None,
                              return_tensor: bool = False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    iso = np.asarray(iso_xyz, dtype=np.float64)
    src_np = beam_source_pos(iso, machine.sad_mm, gantry_deg)
    axis_np, u_hat_np, v_hat_np = beam_basis(gantry_deg)
    density = (density_override.astype(np.float32) if density_override is not None
               else hu_to_density(image.array, machine.hu_anchors))

    src = torch.as_tensor(src_np, dtype=torch.float32, device=dev)
    axis = torch.as_tensor(axis_np, dtype=torch.float32, device=dev)
    u_hat = torch.as_tensor(u_hat_np, dtype=torch.float32, device=dev)
    v_hat = torch.as_tensor(v_hat_np, dtype=torch.float32, device=dev)
    iso_t = torch.as_tensor(iso, dtype=torch.float32, device=dev)
    dens_t = torch.as_tensor(density, dtype=torch.float32, device=dev)

    if coords is None:
        xs = ox + torch.arange(nx, device=dev, dtype=torch.float32) * sx
        ys = oy + torch.arange(ny, device=dev, dtype=torch.float32) * sy
        zs = oz + torch.arange(nz, device=dev, dtype=torch.float32) * sz
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        coords_full = torch.stack([gx, gy, gz], dim=-1)
    else:
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

    bbox = None
    coords_c = coords_full
    if crop_margin is not None:
        bbox = _mask_bbox(open_mask, crop_margin, (nz, ny, nx))
        z0, z1, y0, y1, x0, x1 = bbox
        sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
        open_mask = open_mask[sl]; dens_t = dens_t[sl]
        coords_c = coords_full[(*sl, slice(None))]

    rdepth = radiological_depth_skinentry(density, image.spacing, image.origin, src_np,
                                          axis_np, u_hat_np, v_hat_np, iso,
                                          out_bbox=bbox, coords=coords_full)
    rdepth_t = torch.as_tensor(rdepth, dtype=torch.float32, device=dev)

    fluence = open_mask * torch.exp(-MU_6MV_PER_CM * rdepth_t)
    rel_iso = coords_c - iso_t
    perp = rel_iso - ((rel_iso * axis).sum(-1).unsqueeze(-1) * axis)
    dist_to_cax = torch.linalg.norm(perp, dim=-1)
    source_dist = torch.linalg.norm(coords_c - src, dim=-1)

    result = torch.stack([dens_t, rdepth_t, fluence, dist_to_cax, source_dist], dim=0)
    if return_tensor:
        return result, bbox
    result = result.cpu().numpy().astype(np.float32)
    return result if crop_margin is None else (result, bbox)
