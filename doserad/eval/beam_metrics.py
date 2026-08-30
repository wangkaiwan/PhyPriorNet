"""Beam-level masked MAE (challenge Level-1 metric): mean abs error within the
>=10%-of-max region, normalized by the beam's max GT dose."""
from __future__ import annotations

import numpy as np


def masked_mae(pred: np.ndarray, gt: np.ndarray, hi_frac: float = 0.1) -> float:
    gmax = float(gt.max())
    if gmax <= 0:
        return 0.0
    mask = gt >= hi_frac * gmax
    if not mask.any():
        return 0.0
    return float(np.abs(pred[mask] - gt[mask]).mean() / gmax)


def _idd_curve(d: np.ndarray, axis: np.ndarray, spacing, origin) -> np.ndarray:
    """Sum dose per depth-bin along `axis`. Returns 1D IDD."""
    nz, ny, nx = d.shape
    sx, sy, sz = spacing; ox, oy, oz = origin
    xs = ox + np.arange(nx, dtype=np.float32) * sx
    ys = oy + np.arange(ny, dtype=np.float32) * sy
    zs = oz + np.arange(nz, dtype=np.float32) * sz
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) - np.array(origin)
    depth = (p @ np.asarray(axis, dtype=np.float32)).reshape(d.shape)
    bin_mm = min(spacing)
    dmin, dmax = float(depth.min()), float(depth.max())
    nbin = max(int(np.ceil((dmax - dmin) / bin_mm)), 2)
    idx = np.clip(((depth - dmin) / bin_mm).astype(np.int64), 0, nbin - 1)
    idd = np.bincount(idx.ravel(),
                      weights=d.ravel().astype(np.float64),
                      minlength=nbin)
    return idd.astype(np.float32)


def idd_rms(pred: np.ndarray, gt: np.ndarray,
            axis: np.ndarray, spacing, origin) -> float:
    p_idd = _idd_curve(pred, axis, spacing, origin)
    g_idd = _idd_curve(gt,   axis, spacing, origin)
    peak = float(g_idd.max())
    if peak <= 0:
        return 0.0
    return float(np.sqrt(((p_idd - g_idd) ** 2).mean()) / peak)
