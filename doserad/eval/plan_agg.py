"""Accumulate per-CP dose crops into a full-volume plan dose (plain sum,
matching the dataset convention of no per-CP MU weighting)."""
from __future__ import annotations

import numpy as np


def accumulate_plan(per_cp, full_shape):
    """per_cp: iterable of (arr (d,h,w), bbox (z0,z1,y0,y1,x0,x1) inclusive)."""
    out = np.zeros(full_shape, dtype=np.float32)
    for arr, bbox in per_cp:
        z0, z1, y0, y1, x0, x1 = bbox
        out[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += arr.astype(np.float32)
    return out


def stratified_mae(pred: np.ndarray, gt: np.ndarray, rx: float) -> float:
    """Unweighted mean of MAE in high/mid/low strata defined by gt vs rx."""
    if rx <= 0:
        return 0.0
    pred = pred.astype(np.float32); gt = gt.astype(np.float32)
    strata = [(0.8 * rx, np.inf), (0.3 * rx, 0.8 * rx), (0.1 * rx, 0.3 * rx)]
    vals = []
    for lo, hi in strata:
        m = (gt >= lo) & (gt < hi) if hi != np.inf else (gt >= lo)
        if m.any():
            vals.append(float(np.abs(pred[m] - gt[m]).mean() / rx))
    if not vals:
        return 0.0
    return float(np.mean(vals))
