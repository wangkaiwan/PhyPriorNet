"""Deploy proton inference: geometric bbox -> ray-centric build -> PB-threshold TIGHT crop
(matches the training GT crop: {PB>0.01*max}+4 margin) -> compiled net -> cutoff.

The tight crop is essential: DoseUNet3D's GroupNorm normalizes over spatial dims, so a crop much
larger than the training GT-tight crop shifts the statistics and wrecks the output (gamma 70). The
PB prior (~95% of the dose) is a faithful proxy for the GT dose extent, so a PB-threshold bbox
reproduces the training crop distribution -> gamma restored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
from accel.proton_build_ray import build_ray
from accel.proton_2pass import build_ray_2pass
from container.proton.geom_bbox import geom_bbox_proton

# 2-pass build (default): a cheap coarse-PB pass locates each beamlet's tight bbox, then the full
# WEPL/PB build runs only over that (~4.5x smaller) union instead of the inflated geom box. LOSSLESS
# (gamma abd 98.3=98.3 / lung 97.3=97.3, margin 8), ~2.2x on the build (the 66% deploy component).
# Set DOSERAD_2PASS=0 to fall back to the 1-pass geom-box build.
_USE_2PASS = os.environ.get("DOSERAD_2PASS", "1") != "0"
# ENGINE V2 (DOSERAD_ENGINE_V2=1, default OFF): plan-level batched inference — per-beam BEV-cumsum
# WEPL (388x on build) + pad16-bucketed batched forward. Requires the V3-physics net
# (all75_r2_ft_v3physics: net finetuned on parallel-column WEPL/prior channels). held16-validated
# 2026-08-26: gamma 97.51 == deployed 97.50, 3.1x wall.
_ENGINE_V2 = os.environ.get("DOSERAD_ENGINE_V2", "0") == "1"
_r16 = lambda v: -(-v // 16) * 16

# LUNG DOSE-DEFLATION (DOSERAD_LUNG_DEFLATE=1, default OFF -> byte-identical pipeline).
# 71-pt held-out CV measured a systematic OVER-prediction in lung-HU voxels growing with dose band
# (plan-level b: +0.05% @10-30% max, +0.84% @30-60, +1.51% @60-100; 97% of lung pts over 1 in the
# high band). Gamma is tail-driven and ~unmoved by removing this mean bias (measured −0.08), but the
# DVH clinical score (D98/V95/D2/Dmean) is EXACTLY a mean-dose-error metric — deflating toward GT is
# a DVH-side correction. Applied per-beamlet: multiply lung-HU voxels (CT < -500) by 1/b interpolated
# on the beamlet's own dose fraction (beamlet high band ~= Bragg peak ~= the plan's in-lung high band).
_LUNG_DEFLATE = os.environ.get("DOSERAD_LUNG_DEFLATE", "0") == "1"
_DEFL_CENTERS = np.array([0.06, 0.20, 0.45, 0.805], dtype=np.float32)
# 1/b per band, b fit on ALL 35 lung CV patients (diag_proton_lung_bias.py):
# b = [1.0002, 1.0045, 1.0075, 1.0125] for bands 2-10 / 10-30 / 30-60 / 60-100 % of max.
_DEFL_FACTORS = (1.0 / np.array([1.0002, 1.0045, 1.0075, 1.0125], dtype=np.float32))
_LUNG_HU = -500.0


def _deflate_lung(dose: np.ndarray, ct_hu_crop: np.ndarray) -> np.ndarray:
    dmax = float(dose.max())
    if dmax <= 0:
        return dose
    lung = ct_hu_crop < _LUNG_HU
    if not lung.any():
        return dose
    frac = dose / dmax
    f = np.interp(frac, _DEFL_CENTERS, _DEFL_FACTORS,
                  left=1.0, right=_DEFL_FACTORS[-1]).astype(np.float32)
    out = dose.copy()
    out[lung] = dose[lung] * f[lung]
    return out
_THRESH = 0.01   # matches precompute_proton THRESH
# Margin in voxels around the PB-threshold box (grid is 1x1x3 mm, so 4 = 4 mm in-plane,
# 12 mm axial). Made configurable to test whether proton suffers the same crop truncation
# that cost photon ~7 gamma: photon predicts zero outside its crop while the GT keeps a
# tail there, and our old evaluation was blind to it because the reference was cropped the
# same way. Proton's box follows the dose rather than the aperture, so it may be immune.
_MARGIN = int(os.environ.get("DOSERAD_PROTON_MARGIN", "4"))   # matches precompute_proton


def _tight_from_pb(pb, margin=_MARGIN):
    """pb: (d,h,w) tensor. Return tight bbox indices within the crop, or None."""
    m = pb > _THRESH * float(pb.max())
    nz = torch.nonzero(m, as_tuple=False)
    if nz.numel() == 0:
        return None
    lo = nz.min(0).values.tolist(); hi = nz.max(0).values.tolist()
    d, h, w = pb.shape
    return (max(lo[0] - margin, 0), min(hi[0] + margin, d - 1),
            max(lo[1] - margin, 0), min(hi[1] + margin, h - 1),
            max(lo[2] - margin, 0), min(hi[2] + margin, w - 1))


@torch.no_grad()
def predict_beams(image, beams, density_np, density_t, net, machine, dev="cuda", on_frame=None):
    if _ENGINE_V2:
        from accel.proton_engine_v2 import predict_plan_v2
        return predict_plan_v2(image, beams, density_np, density_t, net, machine, dev,
                               on_frame=on_frame)
    """image: Volume-like (.array (z,y,x), .spacing, .origin). beams: json beams. net: compiled net.
    Returns {(beam_idx,ray_idx,l): (dose_full_bbox_np, tight_bbox_full)} for the container to place.
    `on_frame(key, dose, bbox)`, if given, is called the moment each beamlet is done so the writer
    can start on it while the GPU moves to the next one."""
    pcs = torch.as_tensor(_P_CH_SCALE_PRIOR, device=dev).view(5, 1, 1, 1)
    out = {}
    for b in beams:
        bi = b.get("beam_idx", beams.index(b))
        for ri, r in enumerate(b["rays"]):
            bls = []
            for l, bl in enumerate(r["beamlets"]):
                gb = geom_bbox_proton(density_np, image.spacing, image.origin,
                                      r["ray_source"], r["ray_target"], machine, bl["energy"])
                if gb is None:
                    continue
                bls.append(dict(energy=bl["energy"], bbox=gb, key=(bi, ri, l)))
            if not bls:
                continue
            _build = build_ray_2pass if _USE_2PASS else build_ray
            stacks = _build(image, r["ray_source"], r["ray_target"], bls,
                            machine=machine, density=density_t, device=dev)
            for (stack, gbb), bl in zip(stacks, bls):
                pb = stack[2] * _P_CH_SCALE_PRIOR[2] / PROTON_DOSE_SCALE      # PB back to ~Gy
                tb = _tight_from_pb(pb)
                if tb is None:
                    continue
                z0, z1, y0, y1, x0, x1 = tb
                tight = stack[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
                d, h, w = tight.shape[-3:]
                x = F.pad(tight[None], (0, _r16(w) - w, 0, _r16(h) - h, 0, _r16(d) - d))
                with torch.autocast("cuda"):
                    y = net(x, torch.zeros(1, dtype=torch.long, device=dev))
                dose = (y[0, 0, :d, :h, :w].float() / PROTON_DOSE_SCALE).cpu().numpy().astype(np.float32)
                gz0, _, gy0, _, gx0, _ = gbb
                full_bbox = (gz0 + z0, gz0 + z1, gy0 + y0, gy0 + y1, gx0 + x0, gx0 + x1)
                if _LUNG_DEFLATE:
                    fz0, fz1, fy0, fy1, fx0, fx1 = full_bbox
                    dose = _deflate_lung(dose, image.array[fz0:fz1 + 1, fy0:fy1 + 1, fx0:fx1 + 1])
                if on_frame is not None:
                    on_frame(bl["key"], dose, full_bbox)
                else:
                    out[bl["key"]] = (dose, full_bbox)
    return out
