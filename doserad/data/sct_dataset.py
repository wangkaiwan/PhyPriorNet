"""Paired CT/MRI axial-slice dataset for in-domain sCT training on DoseRAD's
75 patients (mr.mha and ct.mha on the SAME registered grid)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from doserad.io.mha import load_mha


def _normalize_mri(mri: np.ndarray) -> np.ndarray:
    """Body-masked z-score (background MR ~= 0)."""
    body = mri > 0
    if body.any():
        mu = float(mri[body].mean())
        sd = float(mri[body].std()) or 1.0
    else:
        mu, sd = 0.0, 1.0
    return ((mri - mu) / sd).astype(np.float32)


def _normalize_ct(ct: np.ndarray) -> np.ndarray:
    return np.clip(ct / 1000.0, -1.5, 3.0).astype(np.float32)


def _center_crop_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop then zero-pad a (..., H, W) array to (target_h, target_w)."""
    h, w = arr.shape[-2], arr.shape[-1]
    # crop
    if h > target_h:
        y0 = (h - target_h) // 2
        arr = arr[..., y0:y0 + target_h, :]
        h = target_h
    if w > target_w:
        x0 = (w - target_w) // 2
        arr = arr[..., x0:x0 + target_w]
        w = target_w
    # pad
    if h < target_h or w < target_w:
        pad_t = (target_h - h) // 2
        pad_b = target_h - h - pad_t
        pad_l = (target_w - w) // 2
        pad_r = target_w - w - pad_l
        pad_width = [(0, 0)] * (arr.ndim - 2) + [(pad_t, pad_b), (pad_l, pad_r)]
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return arr


class SCTPairedSliceDataset(Dataset):
    def __init__(self, patient_ids, root, slice_context: int = 2,
                 fixed_slice: int | None = None, seed: int = 0,
                 patch_size: int | None = None):
        self.pids = list(patient_ids)
        self.root = Path(root)
        self.k = slice_context
        self.fixed_slice = fixed_slice
        self.rng = np.random.default_rng(seed)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        mr = load_mha(self.root / pid / "image" / "mr.mha").array
        ct = load_mha(self.root / pid / "image" / "ct.mha").array
        mr = _normalize_mri(mr)
        ct = _normalize_ct(ct)
        nz = mr.shape[0]
        z = self.fixed_slice if self.fixed_slice is not None \
            else int(self.rng.integers(nz))
        k = self.k
        stack = np.zeros((2 * k + 1, mr.shape[1], mr.shape[2]), dtype=np.float32)
        for j, zz in enumerate(range(z - k, z + k + 1)):
            if 0 <= zz < nz:
                stack[j] = mr[zz]
        ct_slice = ct[z][None].astype(np.float32)
        if self.patch_size is not None:
            stack = _center_crop_pad(stack, self.patch_size, self.patch_size)
            ct_slice = _center_crop_pad(ct_slice, self.patch_size, self.patch_size)
        return {"mri": stack, "ct": ct_slice, "patient_id": pid}
