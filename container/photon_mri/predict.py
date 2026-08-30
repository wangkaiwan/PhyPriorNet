"""Deploy Photon-MRI inference. Mirrors scripts/eval_dose_e2e.py's per-CP channel assembly EXACTLY
(the validated 91.1 5CV path), but sources the density-independent GEOMETRY channels (dist_to_cax,
source_dist, aperture mask) from photon_channels instead of the training cache — they are identical
by construction (photon_channels made the cache; ch3/ch4/aperture do not depend on density).

Per CP: photon_channels(density_override=synth density, crop_margin=8) -> geometry + aperture bbox
(== training crop, bit-identical, validated for photon-CT); recompute the skin-entry radiological
depth on the synth density; skin-gated naive prior; -> compiled dose net. NOTE the naive here is the
skin-entry naive_dose_torch(skin_gate=True) that the MRI net was trained with — NOT the build-up
naive in inference.pipeline._normalize_gpu (that is the CT-photon container's prior).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from accel.photon_channels_fast import photon_channels_fast
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.diff_channels import fluence_torch, naive_dose_torch
from doserad.physics.diff_channels_skinentry import radiological_depth_skinentry_torch
from doserad.inference.pipeline import _build_coords
from doserad.data.dataset import _CH_SCALE, _NAIVE_SCALE, DOSE_SCALE

# Same knob as the photon-CT path (container/photon/predict.py). It was hardcoded here, so
# baking DOSERAD_PHOTON_MARGIN into the photon-MRI image silently did nothing.
_MARGIN = int(os.environ.get("DOSERAD_PHOTON_MARGIN", "8"))
_pad16 = lambda n: (16 - n % 16) % 16


@torch.no_grad()
def predict_cps(image, beams, density_np, density_t, net_dose, machine, dev="cuda",
                img_ch=False, mr01=None, sct01=None, on_frame=None):
    """image: Volume-like (.array (z,y,x), .spacing, .origin, .sad optional). density_np: synth density
    (z,y,x) cpu. density_t: same on GPU. net_dose: compiled E2E.dose. Returns
    {(beam_idx,cp_idx): (dose_crop_np, bbox)}."""
    coords = _build_coords(image, dev)
    mod = torch.zeros(1, dtype=torch.long, device=dev)
    scale = torch.tensor(list(_CH_SCALE) + [_NAIVE_SCALE], device=dev).view(-1, 1, 1, 1)
    out = {}
    for bi, b in enumerate(beams):
        iso = np.asarray(b.get("iso_center", [0, 0, 0]), np.float64)
        for ci, cp in enumerate(b["control_points"]):
            gantry = cp["gantry_angle"]
            geom, bbox = photon_channels_fast(
                image, machine, iso, gantry,
                np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                dens_t=density_t, coords=coords, crop_margin=_MARGIN, dev=dev)
            z0, z1, y0, y1, x0, x1 = bbox
            sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
            src = beam_source_pos(iso, machine.sad_mm, gantry); ax, u, v = beam_basis(gantry)
            rdepth = radiological_depth_skinentry_torch(density_t, image.spacing, image.origin,
                                                        src, ax, u, v, iso, out_bbox=bbox)
            fl = fluence_torch((geom[2] > 0).float(), rdepth)
            naive = naive_dose_torch(density_t[sl], rdepth, fl, geom[4], skin_gate=True)
            chans = [density_t[sl] / _CH_SCALE[0], rdepth / _CH_SCALE[1], fl / _CH_SCALE[2],
                     geom[3] / _CH_SCALE[3], geom[4] / _CH_SCALE[4], naive / _NAIVE_SCALE]
            if img_ch:
                chans += [mr01[sl], sct01[sl]]
            inp = torch.stack(chans, 0)
            Z, Y, X = inp.shape[-3:]
            inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
            with torch.autocast("cuda"):
                y = net_dose(inp, mod)[0, 0, :Z, :Y, :X].float() / DOSE_SCALE
            key = (b.get("beam_idx", bi), ci)
            dose = y.cpu().numpy().astype(np.float32)
            bb = tuple(int(v_) for v_ in bbox)
            if on_frame is not None:      # stream: the writer starts on this control point while
                on_frame(key, dose, bb)   # the GPU moves on to the next one
            else:
                out[key] = (dose, bb)
    return out
