"""3D local gamma 1%/1mm at >=10% rx — challenge Level-2 metric via PyMedPhys.

Returns pass rate (fraction of evaluation voxels with gamma <= 1).
"""
from __future__ import annotations

import numpy as np
import pymedphys


def gamma_array(pred: np.ndarray, gt: np.ndarray, spacing,
                rx: float, dose_pct: float = 1.0,
                dta_mm: float = 1.0, hi_frac: float = 0.1):
    """Compute the per-voxel local gamma. Returns (gamma, eval_mask) where
    `gamma` has inf where PyMedPhys returned NaN (uncomputed/failed) and
    `eval_mask` = gt >= hi_frac*rx is the evaluation region. Sharing this array
    lets callers derive overall AND dose-band-stratified pass rates from ONE
    (expensive) gamma computation."""
    sx, sy, sz = spacing
    nz, ny, nx = gt.shape
    axes = (np.arange(nz) * sz, np.arange(ny) * sy, np.arange(nx) * sx)
    mask = gt >= hi_frac * rx
    if not mask.any():
        return np.full(gt.shape, np.inf, dtype=np.float64), mask
    g = pymedphys.gamma(axes, gt.astype(np.float64),
                        axes, pred.astype(np.float64),
                        dose_percent_threshold=dose_pct,
                        distance_mm_threshold=dta_mm,
                        local_gamma=True,
                        lower_percent_dose_cutoff=hi_frac * 100.0,
                        max_gamma=2.0)
    g = np.where(np.isnan(g), np.inf, g)
    return g, mask


def gamma_pass(pred: np.ndarray, gt: np.ndarray, spacing,
               rx: float, dose_pct: float = 1.0,
               dta_mm: float = 1.0, hi_frac: float = 0.1) -> float:
    """`spacing` = (sx, sy, sz) mm. gt/pred are (z, y, x) arrays. Local norm."""
    g, mask = gamma_array(pred, gt, spacing, rx, dose_pct, dta_mm, hi_frac)
    if not mask.any():
        return 1.0
    return float((g[mask] <= 1.0).mean())


def gamma_pass_by_band(pred: np.ndarray, gt: np.ndarray, spacing, rx: float,
                       dose_pct: float = 1.0, dta_mm: float = 1.0,
                       hi_frac: float = 0.1) -> dict:
    """Local gamma pass rate overall AND split by GT dose band, to localize
    WHERE gamma fails (the north-star 1%/1mm is hardest in steep gradients).
    Bands (of rx): high ≥80%, mid 30-80%, low 10-30%. Returns pass fractions +
    voxel counts. Uses ONE gamma computation."""
    g, mask = gamma_array(pred, gt, spacing, rx, dose_pct, dta_mm, hi_frac)
    out = {"overall": 1.0 if not mask.any() else float((g[mask] <= 1.0).mean()),
           "n_eval": int(mask.sum())}
    bands = {"high": (0.8 * rx, np.inf), "mid": (0.3 * rx, 0.8 * rx),
             "low": (0.1 * rx, 0.3 * rx)}
    for name, (lo, hi) in bands.items():
        bm = mask & (gt >= lo) & (gt < hi if hi != np.inf else np.ones_like(mask))
        out[name] = float((g[bm] <= 1.0).mean()) if bm.any() else float("nan")
        out[f"n_{name}"] = int(bm.sum())
    return out
