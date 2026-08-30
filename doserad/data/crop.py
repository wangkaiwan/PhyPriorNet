"""Beam bounding-box crop helpers: limit cached channels/dose to the
irradiated region (fluence>0) padded by a margin."""
from __future__ import annotations

import numpy as np

Bbox = tuple[int, int, int, int, int, int]   # z0,z1,y0,y1,x0,x1 inclusive


def compute_bbox(fluence: np.ndarray, margin: int = 8) -> Bbox:
    nz, ny, nx = fluence.shape
    nz_, ny_, nx_ = np.where(fluence > 0)
    if nz_.size == 0:
        return (0, nz - 1, 0, ny - 1, 0, nx - 1)
    z0 = max(int(nz_.min()) - margin, 0); z1 = min(int(nz_.max()) + margin, nz - 1)
    y0 = max(int(ny_.min()) - margin, 0); y1 = min(int(ny_.max()) + margin, ny - 1)
    x0 = max(int(nx_.min()) - margin, 0); x1 = min(int(nx_.max()) + margin, nx - 1)
    return (z0, z1, y0, y1, x0, x1)


def crop_to_bbox(arr: np.ndarray, bbox: Bbox) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return arr[..., z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
