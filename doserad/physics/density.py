"""HU -> mass density (g/cm^3) via the machine's piecewise-linear calibration.
Uses the exact anchor points the Monte-Carlo dose engine used; values outside
the anchor range are clamped to the endpoint densities."""
from __future__ import annotations

import numpy as np


def hu_to_density(hu: np.ndarray,
                  anchors: tuple[tuple[float, float], ...]) -> np.ndarray:
    """`anchors` is a sorted sequence of (hu, density) points."""
    xp = np.asarray([a[0] for a in anchors], dtype=np.float64)
    fp = np.asarray([a[1] for a in anchors], dtype=np.float64)
    rho = np.interp(hu.astype(np.float64), xp, fp)  # np.interp clamps by default
    return rho.astype(np.float32)
