"""Coordinate transforms between the (z,y,x) voxel grid and world (x,y,z) mm.
Assumes identity direction cosines (true for the DoseRAD .mha data)."""
from __future__ import annotations

import numpy as np

from doserad.io.mha import Volume


def voxel_world_coords(vol: Volume) -> np.ndarray:
    """Return an array of shape (z, y, x, 3) giving the world (x, y, z) mm
    coordinate of every voxel center. Identity direction assumed."""
    nz, ny, nx = vol.array.shape
    sx, sy, sz = vol.spacing
    ox, oy, oz = vol.origin
    xs = ox + np.arange(nx, dtype=np.float32) * sx
    ys = oy + np.arange(ny, dtype=np.float32) * sy
    zs = oz + np.arange(nz, dtype=np.float32) * sz
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).astype(np.float32)


# --- Beam / gantry geometry (IEC 61217-like) ---
# Convention: gantry rotates in the WORLD X-Y PLANE.
# GANTRY_SIGN = -1.0 sets the rotation direction (PCA-validated 2026-05-22: beam AXIS
#   aligns with GT dose, median 3.5 deg). BUT PCA only fixes the axis LINE, not which END
#   is the source. SOURCE_SIGN fixes the source SIDE, found 2026-06-11 from the depth-dose:
#   the original side gave an INVERTED PDD (dose rising with depth, corr +0.88); flipping it
#   gives a textbook PDD (dose falls with depth, corr -0.87). The old cache (v1/v2/v3) used
#   the wrong side -> anti-physical fluence -> underfitting.
#   Gantry 0  -> source at iso + (0, -SAD, 0)   Gantry 90 -> source at iso + (+SAD, 0, 0)
GANTRY_SIGN = -1.0
SOURCE_SIGN = -1.0    # source side along the (PCA-validated) axis; -1 gives a physical PDD
BEV_U_SIGN = 1.0      # flip to -1.0 if the MLC-motion axis is mirrored


def beam_source_pos(iso_xyz: np.ndarray, sad_mm: float,
                    gantry_deg: float) -> np.ndarray:
    """World (x,y,z) source position. gantry 0 -> source at iso + (0,-SAD,0)
    (SOURCE_SIGN=-1, validated against the GT depth-dose / PDD)."""
    th = np.deg2rad(GANTRY_SIGN * gantry_deg)
    offset = SOURCE_SIGN * np.array([np.sin(th), np.cos(th), 0.0]) * sad_mm
    return np.asarray(iso_xyz, dtype=np.float64) + offset


def beam_basis(gantry_deg: float):
    """Return (axis, u, v) unit vectors. axis = source->iso beam direction;
    u = MLC-motion (leaf-tip) axis in the iso plane; v = leaf-pair stacking
    axis (world z). axis points from the (corrected) source toward iso."""
    th = np.deg2rad(GANTRY_SIGN * gantry_deg)
    axis = -SOURCE_SIGN * np.array([np.sin(th), np.cos(th), 0.0])
    u = BEV_U_SIGN * np.array([np.cos(th), -np.sin(th), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    return axis, u, v
