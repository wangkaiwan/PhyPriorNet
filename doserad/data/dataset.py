"""Photon-CT dataset over cached beam-bbox crops. Foreground-biased patch
sampling; dose kept in ABSOLUTE dose-to-medium (Gy) scaled by a fixed global
constant; fixed per-channel input scaling.

The challenge scores absolute Gy, and a segment's dose magnitude is determined by
its MLC aperture + patient geometry (both already model inputs), so we train on the
absolute value rather than per-CP-max-normalizing it (which threw the scale away and
left it unrecoverable at inference). DOSE_SCALE only rescales units to O(1) for
stable training; inference divides by the same constant to return Gy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

# Multiply absolute Gy by this to bring per-CP dose (~6.8e-5 Gy median max) to O(1).
DOSE_SCALE = 1.0e4
# foreground = voxels >= this fraction of the crop's own max dose (mirrors the
# challenge masked-MAE metric, which masks voxels >= 10% of beam-max GT).
FG_FRAC = 0.1

# fixed scales for [density, rdepth, fluence, dist_to_cax, source_dist]
_CH_SCALE = np.array([3.0, 30.0, 1.0, 200.0, 1500.0], dtype=np.float32)
_NAIVE_SCALE = 1.0   # naive_dose is already O(1) (fluence*invsq*buildup)
# AAA prior (aaa_prior_dose) raw magnitude is O(10) (kernel·fluence units, per-CP max
# ~10-16 in smoke tests) — divide to bring the channel to O(1). FIXED so the train cache
# and inference compute the IDENTICAL channel (PITFALLS A.3 train/inference consistency).
_AAA_SCALE = 12.0


def normalize_channels(ch: np.ndarray, add_naive: bool = False,
                       scatter: bool = False, aaa: np.ndarray | None = None,
                       naive_skin_gate: float | None = None) -> np.ndarray:
    """Scale the cached 5 physics channels to O(1). If add_naive, append the
    first-order analytical dose as a 6th channel (computed from the raw 5) for
    residual learning (v6+). `scatter=True` (v10) uses the density-scaled
    lateral-scatter prior instead of the scatter-blind one (Tier-1.5). If `aaa` is
    given (precomputed Tier-2 AAA prior, same spatial shape as ch), append it as the
    6th channel instead (v12) — MUST be the identical aaa_prior_dose op at train+infer.
    `add_naive` and `aaa` are mutually exclusive. `naive_skin_gate` (opt-in, default
    False/None = OFF = byte-identical) applies the skin-ENTRY gate to the naive prior:
    zero ONLY the external air upstream of the skin (never masks in-body voxels). Pass
    the SAME value at train+infer (thread via config)."""
    ch = ch.astype(np.float32)
    scale = _CH_SCALE
    if aaa is not None:
        ch = np.concatenate([ch, aaa.astype(np.float32)[None]], axis=0)   # (6, ...)
        scale = np.concatenate([_CH_SCALE, [_AAA_SCALE]]).astype(np.float32)
    elif add_naive:
        from doserad.physics.naive_dose import compute_naive_dose
        naive = compute_naive_dose(ch, scatter=scatter, skin_gate=naive_skin_gate)[None]   # (1, ...)
        ch = np.concatenate([ch, naive], axis=0)             # (6, ...)
        scale = np.concatenate([_CH_SCALE, [_NAIVE_SCALE]]).astype(np.float32)
    return (ch / scale[:, None, None, None]).astype(np.float32)


def _sample_patch(shape, patch, fg_voxels, fg_prob, rng):
    if fg_prob > 0 and len(fg_voxels[0]) > 0 and rng.random() < fg_prob:
        j = rng.integers(len(fg_voxels[0]))
        center = [int(fg_voxels[a][j]) for a in range(3)]
    else:
        center = [int(rng.integers(shape[a])) for a in range(3)]
    return [int(center[a] - patch[a] // 2) for a in range(3)]


def _extract(arr, start, patch):
    """Extract patch from arr (..., z,y,x) at `start`, zero-padding OOB."""
    nz, ny, nx = arr.shape[-3:]
    pz, py, px = patch
    out = np.zeros(arr.shape[:-3] + tuple(patch), dtype=arr.dtype)
    sz, sy, sx = start
    z0, y0, x0 = max(sz, 0), max(sy, 0), max(sx, 0)
    z1, y1, x1 = min(sz + pz, nz), min(sy + py, ny), min(sx + px, nx)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return out
    out[..., z0 - sz:z1 - sz, y0 - sy:y1 - sy, x0 - sx:x1 - sx] = \
        arr[..., z0:z1, y0:y1, x0:x1]
    return out


def _augment(inp, dose, rng, do_rot=True):
    """Physics-exact rigid augmentation: identical axis-aligned flips (all 3
    spatial axes) and optional axial-plane 90 deg rotations applied to BOTH the
    channel stack and the dose. Valid because every channel is a scalar field
    (density, rdepth, fluence, dist_to_cax, source_dist) on an isotropic grid,
    so a rigid transform of the (channels, dose) pair is the dose for the rigidly
    transformed (image, beam) geometry. Spatial axes are the last 3."""
    for ax in (-3, -2, -1):
        if rng.random() < 0.5:
            inp = np.flip(inp, axis=ax)
            dose = np.flip(dose, axis=ax)
    if do_rot:
        k = int(rng.integers(0, 4))
        if k:
            inp = np.rot90(inp, k=k, axes=(-2, -1))
            dose = np.rot90(dose, k=k, axes=(-2, -1))
    return np.ascontiguousarray(inp), np.ascontiguousarray(dose)


class PhotonCropDataset(Dataset):
    def __init__(self, index_rows, cache_dir, patch=(128, 128, 128),
                 fg_prob=0.7, seed=0, add_naive=False,
                 augment=False, augment_rot=True, scatter=False,
                 aaa_cache_dir=None, naive_skin_gate=None):
        self.rows = list(index_rows)
        self.cache = Path(cache_dir)
        self.aaa_cache = Path(aaa_cache_dir) if aaa_cache_dir else None
        self.patch = tuple(patch)
        self.fg_prob = fg_prob
        self.add_naive = add_naive
        self.scatter = scatter
        self.naive_skin_gate = naive_skin_gate
        self.augment = augment
        self.augment_rot = augment_rot
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        stem = f"{r['beam_idx']}_{r['cp_idx']:03d}.npz"
        f = self.cache / r["patient_id"] / stem
        z = np.load(f)
        aaa = None
        if self.aaa_cache is not None:
            aaa = np.load(self.aaa_cache / r["patient_id"] / stem)["aaa"]
        ch = normalize_channels(z["channels"], add_naive=self.add_naive,
                                scatter=self.scatter, aaa=aaa,
                                naive_skin_gate=self.naive_skin_gate)
        d_arr = z["dose"].astype(np.float32)          # absolute dose-to-medium (Gy)
        dmax = float(d_arr.max()) or 1.0
        dscaled = d_arr * DOSE_SCALE
        fg = np.where(d_arr > FG_FRAC * dmax)
        start = _sample_patch(d_arr.shape, self.patch, fg, self.fg_prob, self.rng)
        inp = _extract(ch, start, self.patch).astype(np.float32)
        dpatch = _extract(dscaled, start, self.patch).astype(np.float32)[None]
        if self.augment:
            inp, dpatch = _augment(inp, dpatch, self.rng, self.augment_rot)
        return {"input": inp, "dose": dpatch, "modality": 0,
                "dose_max": float(z["dose_max"]), "patient_id": r["patient_id"],
                "beam_idx": int(r["beam_idx"]), "cp_idx": int(r["cp_idx"])}
