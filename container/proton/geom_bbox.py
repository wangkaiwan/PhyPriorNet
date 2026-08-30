"""Deploy-time crop bbox from beam GEOMETRY (no GT dose available).

A proton beamlet deposits dose along its ray from the skin to ~skin+range, within a few sigma
laterally. We march the central ray, keep the segment inside the body (density>skin_thr) up to a
generous geometric depth, and take the voxel bbox of that segment expanded by a lateral margin.
This is a SUPERSET of the nonzero dose region (correctness needs superset, not tightness).
"""
from __future__ import annotations

import numpy as np


def geom_bbox_proton(density, spacing, origin, ray_source, ray_target, machine, energy,
                     lateral_mm: float = 24.0, skin_thr: float = 0.05, step_mm: float = 2.0):
    """density: (z,y,x) np. spacing/origin: x-first (sx,sy,sz)/(ox,oy,oz). Returns
    (z0,z1,y0,y1,x0,x1) inclusive voxel bbox, clamped to the volume."""
    nz, ny, nx = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin
    tgt = np.asarray(ray_target, np.float64); jsrc = np.asarray(ray_source, np.float64)
    axis = tgt - jsrc; axis /= (np.linalg.norm(axis) + 1e-12)
    src = tgt - axis * machine.sad

    # generous geometric range: WEPL depth_max / a low-density floor so we never clip distal dose
    # in lung/heterogeneous rays (tighter central-ray-range estimation was measured to clip -> lower gamma).
    eidx = machine.energy_index(energy)
    n = int(machine.lengths[eidx].item())
    depth_max_mm = float(machine.depths[eidx, :n].max().item() + machine.offset[eidx].item())
    geom_range = depth_max_mm / 0.45

    dist_src_tgt = np.linalg.norm(tgt - src)
    ts = np.arange(max(dist_src_tgt - 300.0, 0.0), dist_src_tgt + geom_range, step_mm)
    pts = src[None, :] + ts[:, None] * axis[None, :]
    vx = np.round((pts[:, 0] - ox) / sx).astype(int)
    vy = np.round((pts[:, 1] - oy) / sy).astype(int)
    vz = np.round((pts[:, 2] - oz) / sz).astype(int)
    inb = (vx >= 0) & (vx < nx) & (vy >= 0) & (vy < ny) & (vz >= 0) & (vz < nz)
    vx, vy, vz = vx[inb], vy[inb], vz[inb]
    if len(vx) == 0:
        return None
    body = density[vz, vy, vx] > skin_thr
    if body.any():
        first = int(np.argmax(body))
        last = len(body) - 1 - int(np.argmax(body[::-1]))
        last = min(last + int(20 / step_mm), len(body) - 1)
        vx, vy, vz = vx[first:last + 1], vy[first:last + 1], vz[first:last + 1]
    mx = int(np.ceil(lateral_mm / min(sx, sy, sz)))
    x0 = max(int(vx.min()) - mx, 0); x1 = min(int(vx.max()) + mx, nx - 1)
    y0 = max(int(vy.min()) - mx, 0); y1 = min(int(vy.max()) + mx, ny - 1)
    z0 = max(int(vz.min()) - mx, 0); z1 = min(int(vz.max()) + mx, nz - 1)
    return (z0, z1, y0, y1, x0, x1)
