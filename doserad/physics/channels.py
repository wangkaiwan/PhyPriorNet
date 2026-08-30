"""Assemble the photon input channel stack for one (image, control point):
[density, radiological_depth, primary_fluence, dist_to_cax, source_dist].
Shape (5, z, y, x), float32, on the image grid."""
from __future__ import annotations

import numpy as np
import torch

from doserad.io.mha import Volume
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import PhotonMachine
from doserad.physics.raytrace import radiological_depth_fast

MU_6MV_PER_CM = 0.0475   # effective linear attenuation (1/cm) in water at 6 MV


def _mask_bbox(open_mask: torch.Tensor, margin: int, shape) -> tuple:
    """Inclusive (z0,z1,y0,y1,x0,x1) bbox of open_mask>0, padded by `margin`."""
    nz, ny, nx = shape
    idx = torch.nonzero(open_mask > 0, as_tuple=False)
    if idx.numel() == 0:
        return (0, nz - 1, 0, ny - 1, 0, nx - 1)
    mn = idx.min(0).values.tolist()
    mx = idx.max(0).values.tolist()
    return (max(mn[0] - margin, 0), min(mx[0] + margin, nz - 1),
            max(mn[1] - margin, 0), min(mx[1] + margin, ny - 1),
            max(mn[2] - margin, 0), min(mx[2] + margin, nx - 1))


def photon_channels(image: Volume, machine: PhotonMachine,
                    iso_xyz, gantry_deg: float,
                    mlc_left: np.ndarray, mlc_right: np.ndarray,
                    *, density_override: np.ndarray | None = None,
                    crop_margin: int | None = None,
                    coords: "torch.Tensor | None" = None,
                    return_tensor: bool = False):
    """Assemble the (5, z, y, x) channel stack on the image grid.

    Default (`crop_margin=None`) returns the full-volume stack (precompute/training
    path — unchanged). When `crop_margin` is an int, returns `(cropped_stack, bbox)`
    where the stack is restricted to the beam bounding box (open aperture + margin):
    the runtime fast-path. The expensive radiological-depth sampling and per-voxel
    channel math run only inside the bbox, while the BEV fan stays full-volume, so the
    cropped stack is bit-identical to cropping the full stack to the same bbox."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    iso = np.asarray(iso_xyz, dtype=np.float64)
    src_np = beam_source_pos(iso, machine.sad_mm, gantry_deg)
    axis_np, u_hat_np, v_hat_np = beam_basis(gantry_deg)
    if density_override is not None:
        density = density_override.astype(np.float32)
    else:
        density = hu_to_density(image.array, machine.hu_anchors)  # (z,y,x)

    # Move all geometry to GPU for the O(volume) operations
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
        coords_full = torch.stack([gx, gy, gz], dim=-1)      # (z,y,x,3) world
    else:
        coords_full = coords                                 # per-image reuse

    # project each voxel back to the iso plane along source->voxel -> (u,v) mm
    vec = coords_full - src                                   # (z,y,x,3)
    denom = (vec * axis).sum(-1)
    t_iso = ((iso_t - src) * axis).sum() / torch.where(
        denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    hit = src + t_iso.unsqueeze(-1) * vec                    # iso-plane points
    rel = hit - iso_t
    u = (rel * u_hat).sum(-1)                                # (z,y,x) mm
    v = (rel * v_hat).sum(-1)

    half = machine.num_leaf_pairs / 2.0
    pair = torch.floor(v / machine.leaf_thickness_mm + half).long()
    valid = (pair >= 0) & (pair < machine.num_leaf_pairs)
    pidx = pair.clamp(0, machine.num_leaf_pairs - 1)
    ml_t = torch.as_tensor(mlc_left, dtype=torch.float32, device=dev)
    mr_t = torch.as_tensor(mlc_right, dtype=torch.float32, device=dev)
    left_t = ml_t[pidx]
    right_t = mr_t[pidx]
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
        open_mask = open_mask[sl]
        dens_t = dens_t[sl]
        coords_c = coords_full[(*sl, slice(None))]

    # radiological depth: full grid, or only the bbox subgrid (BEV fan stays full vol)
    rdepth = radiological_depth_fast(density, image.spacing, image.origin, src_np,
                                     axis_np, u_hat_np, v_hat_np, iso,
                                     out_bbox=bbox, coords=coords_full)  # g/cm^2
    rdepth_t = torch.as_tensor(rdepth, dtype=torch.float32, device=dev)

    fluence = open_mask * torch.exp(-MU_6MV_PER_CM * rdepth_t)
    rel_iso = coords_c - iso_t
    perp = rel_iso - ((rel_iso * axis).sum(-1).unsqueeze(-1) * axis)
    dist_to_cax = torch.linalg.norm(perp, dim=-1)
    source_dist = torch.linalg.norm(coords_c - src, dim=-1)

    result = torch.stack([dens_t, rdepth_t, fluence, dist_to_cax, source_dist], dim=0)
    if return_tensor:
        return result, bbox                          # GPU tensor (inference fast-path)
    result = result.cpu().numpy().astype(np.float32)
    return result if crop_margin is None else (result, bbox)
