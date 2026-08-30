"""MLC aperture test in BEV isocenter-plane coordinates (u along leaf motion,
v along leaf-pair stacking), all in mm. Leaf pair i covers
v in [(i - n/2)*t, (i - n/2 + 1)*t)."""
from __future__ import annotations

import numpy as np


def _pair_index(v: float, n_pairs: int, leaf_thickness: float) -> int:
    half = n_pairs / 2.0
    idx = int(np.floor(v / leaf_thickness + half))
    return idx


def aperture_open(u: float, v: float,
                  left_tips: np.ndarray, right_tips: np.ndarray,
                  n_pairs: int, leaf_thickness: float,
                  jaw_x: tuple[float, float],
                  jaw_y: tuple[float, float]) -> bool:
    if not (jaw_x[0] <= u <= jaw_x[1] and jaw_y[0] <= v <= jaw_y[1]):
        return False
    i = _pair_index(v, n_pairs, leaf_thickness)
    if i < 0 or i >= n_pairs:
        return False
    # A zero-width aperture (left == right) is treated as closed.
    return bool(left_tips[i] < right_tips[i] and left_tips[i] <= u <= right_tips[i])
