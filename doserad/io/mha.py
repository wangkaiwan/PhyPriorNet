"""Thin wrappers around SimpleITK for loading/saving .mha volumes as numpy
arrays while preserving spatial metadata. Array axis order is (z, y, x).

Note: spacing/origin follow SimpleITK's x-first convention, which is the reverse
of the numpy array axis order (z, y, x). spacing[0] is the x (column) voxel size,
while array.shape[0] is the z (slice) dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class Volume:
    array: np.ndarray            # (z, y, x)
    spacing: tuple               # (sx, sy, sz) mm — x-first (SimpleITK); array axis 0 is z
    origin: tuple                # (ox, oy, oz) mm — x-first (SimpleITK); array axis 0 is z
    direction: tuple             # 9-tuple row-major

    @property
    def shape(self) -> tuple:
        return self.array.shape


def load_mha(path: str | Path) -> Volume:
    img = sitk.ReadImage(str(path))
    return Volume(
        array=sitk.GetArrayFromImage(img).astype(np.float32),
        spacing=tuple(float(s) for s in img.GetSpacing()),
        origin=tuple(float(o) for o in img.GetOrigin()),
        direction=tuple(float(d) for d in img.GetDirection()),
    )


def save_mha(vol: Volume, path: str | Path) -> None:
    img = sitk.GetImageFromArray(vol.array.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in vol.spacing))
    img.SetOrigin(tuple(float(o) for o in vol.origin))
    img.SetDirection(tuple(float(d) for d in vol.direction))
    sitk.WriteImage(img, str(path))
