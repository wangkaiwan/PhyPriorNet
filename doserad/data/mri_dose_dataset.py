"""MRI->dose dataset for the two ablations (NEW file; does NOT touch the CT-photon pipeline).
Reuses the SAME fold crop cache (bbox/dose/geometry) + fold_0 split as the CT dose model, but
replaces the CT-density-derived channels with MRI (+ optional sCT). Read-only imports from the
production dataset (DOSE_SCALE, FG_FRAC, _CH_SCALE, _sample_patch, _extract).

Channels:
  exp1  ('mri'               ): [MRI, dist, source, open_mask]                      in_ch=4
  exp2  ('mri_sct'           ): [MRI, sCT, dist, source, open_mask]                 in_ch=5
  exp3a ('mri_sct_phys'      ): [MRI, sCT, density, rdepth, fluence, dist, source]  in_ch=7
  exp3b ('mri_sct_phys_naive'): exp3a + naive_prior                                 in_ch=8
geometry channels (dist=ch3, source=ch4) come straight from the cached crop; open_mask=(fluence>0).
exp3 adds the v13 physics channels (density/rdepth/fluence[/naive]) recomputed from the **sCT**
(treated as CT — no GT-CT leakage), read from the precomputed `photon_sct_v4` crop cache
(sct_phys_dir; key 'sct_phys' = [density, rdepth, fluence], same bbox as the real crop).
MRI normalized per-patient 1-99% -> [0,1]; sCT HU window [-1000,2000] -> [0,1].
Also returns GT density (g/cm3, from ct.mha, training-only) for the het/lung loss weighting."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from doserad.data.dataset import DOSE_SCALE, FG_FRAC, _CH_SCALE, _NAIVE_SCALE, _sample_patch, _extract
from doserad.physics.density import hu_to_density
from doserad.physics.naive_dose import compute_naive_dose

CT_MIN, CT_MAX = -1000.0, 2000.0   # sCT HU window (matches the paired sCT trainer)
_SCT_MODES = ("mri_sct", "mri_sct_phys", "mri_sct_phys_naive")   # modes that load the raw sCT image


def _arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)   # (z,y,x)


class MRIDoseDataset(Dataset):
    def __init__(self, rows, cache_dir, root_dir, hu_anchors, patch=(128, 128, 128),
                 mode="mri", sct_dir=None, sct_phys_dir=None, fg_prob=0.7, seed=0):
        assert mode in ("mri", "mri_sct", "mri_sct_phys", "mri_sct_phys_naive")
        self.rows = list(rows)
        self.cache = Path(cache_dir)
        self.root = Path(root_dir)
        self.hu_anchors = hu_anchors
        self.patch = tuple(patch)
        self.mode = mode
        self.sct_dir = Path(sct_dir) if sct_dir else None
        self.sct_phys_dir = Path(sct_phys_dir) if sct_phys_dir else None   # exp3 sct-derived channels
        if mode in ("mri_sct_phys", "mri_sct_phys_naive"):
            assert self.sct_phys_dir is not None, "exp3 modes need sct_phys_dir"
        self.fg_prob = fg_prob
        self.rng = np.random.default_rng(seed)
        self._vc = {}

    def __len__(self):
        return len(self.rows)

    def _vol(self, pid):
        if pid in self._vc:
            return self._vc[pid]
        mr = _arr(self.root / pid / "image" / "mr.mha")
        lo, hi = np.percentile(mr, 1), np.percentile(mr, 99)
        if hi <= lo:
            hi = lo + 1.0
        mr01 = np.clip((mr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        dens = hu_to_density(_arr(self.root / pid / "image" / "ct.mha"), self.hu_anchors).astype(np.float32)
        sct01 = None
        if self.mode in _SCT_MODES:
            sct = _arr(self.sct_dir / pid / "sCT.mha")
            sct01 = np.clip((sct - CT_MIN) / (CT_MAX - CT_MIN), 0.0, 1.0).astype(np.float32)
        # cache as float16 to halve RAM (these are normalised [0,1]/density volumes; upcast at crop
        # time). The per-patient volume cache is otherwise unbounded -> OOM with multiple workers.
        self._vc[pid] = (mr01.astype(np.float16), dens.astype(np.float16),
                         None if sct01 is None else sct01.astype(np.float16))
        return self._vc[pid]

    def __getitem__(self, i):
        r = self.rows[i]
        pid = r["patient_id"]
        z = np.load(self.cache / pid / f"{r['beam_idx']}_{r['cp_idx']:03d}.npz")
        ch = z["channels"].astype(np.float32)            # (5,z,y,x) on bbox crop
        bbox = [int(v) for v in z["bbox"]]
        d_arr = z["dose"].astype(np.float32)             # (z,y,x) Gy on bbox crop
        z0, z1, y0, y1, x0, x1 = bbox                     # INCLUSIVE bbox -> slice z0:z1+1
        sz = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))

        dist = ch[3] / float(_CH_SCALE[3])
        source = ch[4] / float(_CH_SCALE[4])
        open_mask = (ch[2] > 0).astype(np.float32)
        mr01, dens, sct01 = self._vol(pid)
        mr_c = mr01[sz]
        dens_c = dens[sz]
        if self.mode in ("mri_sct_phys", "mri_sct_phys_naive"):
            # exp3: v13 physics channels recomputed from sCT (density, rdepth, fluence) + geometry
            sp = np.load(self.sct_phys_dir / pid / f"{r['beam_idx']}_{r['cp_idx']:03d}.npz")["sct_phys"].astype(np.float32)
            d_s, rd_s, fl_s = sp[0], sp[1], sp[2]
            chans = [mr_c, sct01[sz],
                     d_s / float(_CH_SCALE[0]), rd_s / float(_CH_SCALE[1]), fl_s / float(_CH_SCALE[2]),
                     dist, source]
            if self.mode == "mri_sct_phys_naive":
                # naive prior from the sCT physics (5-ch [density,rdepth,fluence,dist,source], raw units)
                full5 = np.stack([d_s, rd_s, fl_s, ch[3], ch[4]], 0)
                naive = compute_naive_dose(full5).astype(np.float32)
                chans.append(naive / float(_NAIVE_SCALE))
        else:
            chans = [mr_c]
            if self.mode == "mri_sct":
                chans.append(sct01[sz])
            chans += [dist, source, open_mask]
        inp = np.stack(chans, 0).astype(np.float32)      # (C,z,y,x)

        dmax = float(d_arr.max()) or 1.0
        fg = np.where(d_arr > FG_FRAC * dmax)
        start = _sample_patch(d_arr.shape, self.patch, fg, self.fg_prob, self.rng)
        inp_p = _extract(inp, start, self.patch).astype(np.float32)
        dose_p = _extract(d_arr * DOSE_SCALE, start, self.patch).astype(np.float32)[None]
        dens_p = _extract(dens_c, start, self.patch).astype(np.float32)[None]
        return {"input": inp_p, "dose": dose_p, "density": dens_p, "modality": 0,
                "patient_id": pid}
