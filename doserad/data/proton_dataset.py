"""Proton per-beamlet dose dataset (Phase-2). NEW file; reuses photon patch helpers
(_sample_patch, _extract, FG_FRAC) read-only. Loads the no-prior channels cache
(precompute_proton.py) and, if `prior_dir` is given, the SEPARATE pyRadPlan PB-prior
cache (precompute_proton_prior.py) → inserts pb_prior as channel 2 (the v13-style
prior A/B: in_ch4 no-prior vs in_ch5 with-prior).

Cached npz (no-prior): channels(4)=[density(g/cm^3), WEPL(g/cm^2), lateral_dist(mm),
energy(/250)] fp16, dose(d,h,w) Gy fp16, bbox, energy, dose_max.
Prior npz: pb_prior(d,h,w) Gy fp16 on the SAME bbox.

Network input is normalised by _P_CH_SCALE; dose by PROTON_DOSE_SCALE. Returns GT density
(channel 0, raw g/cm^3) separately for the het/lung weighted_l1 (training-only).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from doserad.data.dataset import FG_FRAC, _sample_patch, _extract

PROTON_DOSE_SCALE = 1.0e3          # proton beamlet dose ~1e-3 Gy -> O(1)
# per-channel normalisation for [density, WEPL, (pb_prior), lateral_dist, energy]
_P_CH_SCALE = np.array([2.0, 30.0, 200.0, 1.0], dtype=np.float32)        # no-prior (4ch)
_P_CH_SCALE_PRIOR = np.array([2.0, 30.0, 1.0, 200.0, 1.0], dtype=np.float32)  # ch2=pb_prior (pre-scaled by DOSE_SCALE)


class ProtonDoseDataset(Dataset):
    def __init__(self, rows, cache_dir, prior_dir=None, wepl_dir=None, patch=(32, 128, 128),
                 fg_prob=0.7, seed=0):
        self.rows = list(rows)            # each: {patient_id, fname (B*_R*_L*.npz)}
        self.cache = Path(cache_dir)
        self.prior = Path(prior_dir) if prior_dir else None
        self.wepl = Path(wepl_dir) if wepl_dir else None    # task3: corrected WEPL override (ch1)
        self.patch = tuple(patch)
        self.fg_prob = fg_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        z = np.load(self.cache / r["patient_id"] / r["fname"])
        ch = z["channels"].astype(np.float32)             # (4,d,h,w) raw
        if self.wepl is not None:                         # task3: replace BEV-fan WEPL (ch1) with corrected ray-march WEPL
            ch = ch.copy()
            ch[1] = np.load(self.wepl / r["patient_id"] / r["fname"])["wepl"].astype(np.float32)
        dose = z["dose"].astype(np.float32)               # (d,h,w) Gy
        density = ch[0].copy()                             # raw g/cm^3 for weighting
        if self.prior is not None:
            pb = np.load(self.prior / r["patient_id"] / r["fname"])["pb_prior"].astype(np.float32)
            inp = np.stack([ch[0], ch[1], pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0)
            inp = inp / _P_CH_SCALE_PRIOR[:, None, None, None]
        else:
            inp = ch / _P_CH_SCALE[:, None, None, None]

        dmax = float(dose.max()) or 1.0
        fg = np.where(dose > FG_FRAC * dmax)
        start = _sample_patch(dose.shape, self.patch, fg, self.fg_prob, self.rng)
        inp_p = _extract(inp, start, self.patch).astype(np.float32)
        dose_p = _extract(dose * PROTON_DOSE_SCALE, start, self.patch).astype(np.float32)[None]
        dens_p = _extract(density, start, self.patch).astype(np.float32)[None]
        return {"input": inp_p, "dose": dose_p, "density": dens_p, "modality": 0,
                "patient_id": r["patient_id"]}


def build_proton_rows(cache_dir, patient_ids):
    """List (patient_id, fname) for every cached beamlet of the given patients."""
    cache = Path(cache_dir)
    rows = []
    for pid in patient_ids:
        d = cache / pid
        if not d.exists():
            continue
        for f in sorted(d.glob("B*_R*_L*.npz")):
            if ".tmp" in f.name:
                continue
            rows.append({"patient_id": pid, "fname": f.name})
    return rows
