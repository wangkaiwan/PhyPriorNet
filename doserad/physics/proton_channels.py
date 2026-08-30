"""Proton per-beamlet input channel stack (Phase-2, Task3/4). NEW file — the CT-photon
pipeline is only REUSED (hu_to_density, radiological_depth_fast), never modified.

Mirrors the photon v6/v13 recipe (analytical prior -> residual DL). For ONE beamlet
(ray_source -> ray_target pencil, given energy) on a tight per-beamlet bbox crop:

    [density, WEPL, (pb_prior), lateral_dist, energy]                 (in_ch 4 or 5)

  - density     g/cm^3 from CT HU (hu_to_density), same as photon ch0.
  - WEPL        water-equiv path length g/cm^2 = density integral source->voxel; the proton
                analog of photon radiological_depth → REUSES radiological_depth_fast with
                source=ray_source. Sets the Bragg-peak position.
  - pb_prior    OPTIONAL analytical pyRadPlan Hong PB dose (the GT-aligned "naive proton
                dose"; corr 0.966 vs MC GT). Passed in from the precomputed cache; the net
                learns the RESIDUAL. (omit -> in_ch 4, the no-prior baseline.)
  - lateral_dist  mm perpendicular distance from voxel to the pencil central axis.
  - energy      beamlet energy (MeV) / _E_SCALE, broadcast over the crop (range encoding).

Proton dose is very sparse (~0.04% nonzero, bbox ~20x35x55) → tight bbox crops, great for runtime.
"""
from __future__ import annotations
import numpy as np
import torch

from doserad.physics.density import hu_to_density
from doserad.physics.raytrace import radiological_depth_fast

_E_SCALE = 250.0   # MeV; clinical proton range ~31-201 MeV here -> ~[0.13, 0.80]


def _perp_basis(axis: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to `axis` (unit)."""
    a = axis / (np.linalg.norm(axis) + 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(a, u);   v /= (np.linalg.norm(v) + 1e-9)
    return u.astype(np.float32), v.astype(np.float32)


def proton_channels(image, ray_source, ray_target, energy, hu_anchors, *,
                    out_bbox, pb_prior=None, density_override=None,
                    device: str | None = None, return_tensor: bool = False):
    """Assemble the proton channel stack on the bbox crop `out_bbox`=(z0,z1,y0,y1,x0,x1) (inclusive).

    `image` is a Volume (array (z,y,x), spacing (sx,sy,sz), origin (ox,oy,oz)).
    `pb_prior` (optional) = analytical PB dose on the SAME crop (dz,dy,dx) or None.
    Returns (stack_np (C,dz,dy,dx), bbox) — or (gpu_tensor, bbox) if return_tensor.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nz, ny, nx = image.array.shape
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    density = (density_override if density_override is not None
               else hu_to_density(image.array, hu_anchors)).astype(np.float32)

    src = np.asarray(ray_source, dtype=np.float32)
    tgt = np.asarray(ray_target, dtype=np.float32)
    axis = (tgt - src); axis = (axis / (np.linalg.norm(axis) + 1e-9)).astype(np.float32)
    u_hat, v_hat = _perp_basis(axis)

    # WEPL (g/cm^2): density integral source->voxel, restricted to the crop (BEV fan full-vol).
    wepl = radiological_depth_fast(density, image.spacing, image.origin, src,
                                   axis, u_hat, v_hat, tgt, out_bbox=out_bbox,
                                   device=dev)                       # (dz,dy,dx)

    z0, z1, y0, y1, x0, x1 = out_bbox
    sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))

    # crop coords (world) for lateral distance to the pencil axis
    xs = ox + torch.arange(x0, x1 + 1, device=dev, dtype=torch.float32) * sx
    ys = oy + torch.arange(y0, y1 + 1, device=dev, dtype=torch.float32) * sy
    zs = oz + torch.arange(z0, z1 + 1, device=dev, dtype=torch.float32) * sz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1)                      # (dz,dy,dx,3)
    src_t = torch.as_tensor(src, device=dev); axis_t = torch.as_tensor(axis, device=dev)
    rel = coords - src_t
    along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
    lateral_dist = torch.linalg.norm(rel - along, dim=-1)           # mm (dz,dy,dx)

    dens_t = torch.as_tensor(density[sl], device=dev)
    wepl_t = torch.as_tensor(wepl, device=dev)
    energy_t = torch.full_like(dens_t, float(energy) / _E_SCALE)

    chans = [dens_t, wepl_t]
    if pb_prior is not None:
        chans.append(torch.as_tensor(np.asarray(pb_prior, dtype=np.float32), device=dev))
    chans += [lateral_dist, energy_t]
    stack = torch.stack(chans, dim=0)
    if return_tensor:
        return stack, out_bbox
    return stack.cpu().numpy().astype(np.float32), out_bbox
