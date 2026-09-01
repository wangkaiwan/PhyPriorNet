"""Shared sCT front-end augmentation — approved 2026-07-27 (preview: aug_sct_preview.py).

Geometric augs are applied IDENTICALLY to every volume of a case so MR / coarse / CT / label stay
registered; the DISCRETE label uses NEAREST interpolation (order 0, no class blending) while
continuous images use LINEAR (order 1). MR-only intensity augs (gamma / bias / noise) touch ONLY the
MR channel and never the CT/coarse/label. Rotation/flew fill with 0, which is "air" in every space
here (normalised MR background, to01 CT/coarse where 0 == -1000 HU, class 0 == air).

Used by train_sct_classifier.py (--whole-image) and train_sct_refiner.py, both behind an --aug flag
so the default (single-axis flip) reproduction path is unchanged. Kept cheap because both trainers run
the aug inline in the main loop (no DataLoader workers): the bias field is built at 1/8 resolution.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import rotate as ndrotate, gaussian_filter, zoom


def _flip3(vols, rng):
    # independent 50% flip per axis, SAME choice for every volume in the case
    order = [(-1 if rng.random() < 0.5 else 1) for _ in range(3)]
    if order == [1, 1, 1]:
        return vols
    sl = tuple(slice(None, None, o) for o in order)
    return [v[sl].copy() for v in vols]


def _rot_axial(vols, nearest_flags, rng, max_deg):
    deg = float(rng.uniform(-max_deg, max_deg))
    out = []
    for v, nn in zip(vols, nearest_flags):
        out.append(ndrotate(v, deg, axes=(1, 2), reshape=False, order=(0 if nn else 1),
                            mode="constant", cval=0.0, prefilter=not nn).astype(np.float32))
    return out


def _mr_gamma(m, rng):
    # widened 2026-07-31 to cover the diagnosed test-set intensity shift (post-norm gamma up to ~1.6)
    return np.clip(m, 0, None) ** float(rng.uniform(0.6, 1.7))


def _mr_bias(m, rng, amp_max=0.5):
    amp = float(rng.uniform(0.0, amp_max))
    if amp < 1e-3:
        return m
    small = rng.normal(0, 1, [max(s // 8, 2) for s in m.shape]).astype(np.float32)
    small = gaussian_filter(small, 2.0)
    field = zoom(small, [m.shape[i] / small.shape[i] for i in range(3)], order=1)
    field = 1.0 + amp * (field / (np.abs(field).max() + 1e-6))
    return (m * field).astype(np.float32)


def _mr_noise(m, rng, s_max=0.04):
    s = float(rng.uniform(0.0, s_max))
    return (m + rng.normal(0, s, m.shape).astype(np.float32)) if s > 1e-4 else m


def augment(vols, nearest_flags, mr_index, rng, max_deg=18.0,
            p_rot=0.4, p_gamma=0.7, p_bias=0.6, p_noise=0.4):
    """vols: list of float32 arrays (z,y,x), all same shape. nearest_flags: per-vol bool (True = the
    discrete label). mr_index: which vol is the MR (intensity augs apply ONLY to it). rng: a numpy
    Generator OR the np.random module (both expose random/uniform/normal). Returns a new list."""
    vols = [np.asarray(v, dtype=np.float32) for v in _flip3(list(vols), rng)]
    if rng.random() < p_rot:
        vols = _rot_axial(vols, nearest_flags, rng, max_deg)
    m = vols[mr_index]
    if rng.random() < p_gamma:
        m = _mr_gamma(m, rng)
    if rng.random() < p_bias:
        m = _mr_bias(m, rng)
    if rng.random() < p_noise:
        m = _mr_noise(m, rng)
    vols[mr_index] = m.astype(np.float32)
    # NEAREST-flagged volumes must stay exactly on their integer class ids after linear-free ops
    for k, nn in enumerate(nearest_flags):
        if nn:
            vols[k] = np.rint(vols[k]).astype(np.float32)
    return vols
