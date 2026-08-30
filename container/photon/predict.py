"""Deploy photon inference: per control point, photon_channels(crop_margin=8) -> normalize (+naive
skin-entry prior) -> compiled net -> dose crop. The training crop bbox = compute_bbox(fluence)+8,
and photon_channels(crop_margin=8) uses the same aperture (fluence>0) mask, so the deploy crop is
bit-identical to the training crop — NO GT bbox problem (unlike proton).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doserad.physics.channels import photon_channels
from doserad.inference.pipeline import _normalize_gpu, _build_coords
from doserad.eval.plan_predict import pad_to_multiple
from doserad.data.dataset import DOSE_SCALE
from accel.photon_channels_fast import photon_channels_fast

# Crop margin around the MLC aperture, in voxels (2 mm each). The net is fully convolutional and
# learns a residual on analytic physics channels, so a wider margin costs compute but needs no
# retraining. It matters because a photon control point has broad scatter tails: 4% of the plan's
# energy falls outside margin-8 crops, concentrated in the 10-20%-of-max band where a LOCAL 1%
# gamma has almost no tolerance. Scoring against equally-cropped cached GT hid this entirely
# (96.18 vs 86.38 against the raw full-grid GT on 1ABB006).
_MARGIN = int(os.environ.get("DOSERAD_PHOTON_MARGIN", "8"))
# GPU-resident channel builder (bit-identical, 6.2x on channels). DOSERAD_PHOTON_FAST=0 falls back.
_USE_FAST = os.environ.get("DOSERAD_PHOTON_FAST", "1") != "0"


@torch.no_grad()
def predict_cps(image, beams, density_np, net, machine, dev="cuda", add_naive=True,
                on_frame=None):
    """image: Volume-like (.array (z,y,x), .spacing, .origin). beams: json photon beams (each with
    control_points). Returns {(beam_idx,cp_idx): (dose_crop_np, bbox)}."""
    coords = _build_coords(image, dev)
    dens_t = torch.as_tensor(density_np, dtype=torch.float32, device=dev)   # upload ONCE (not per CP)
    out = {}
    for bi, b in enumerate(beams):
        iso = b.get("iso_center", [0, 0, 0])
        for ci, cp in enumerate(b["control_points"]):
            if _USE_FAST:
                crop, bbox = photon_channels_fast(
                    image, machine, iso, cp["gantry_angle"],
                    np.asarray(cp["mlc_left_int_mm"]), np.asarray(cp["mlc_right_int_mm"]),
                    dens_t=dens_t, coords=coords, crop_margin=_MARGIN, dev=dev)
            else:
                crop, bbox = photon_channels(
                    image=image, machine=machine, iso_xyz=iso, gantry_deg=cp["gantry_angle"],
                    mlc_left=np.asarray(cp["mlc_left_int_mm"]), mlc_right=np.asarray(cp["mlc_right_int_mm"]),
                    density_override=density_np, coords=coords, crop_margin=_MARGIN, return_tensor=True)
            crop = _normalize_gpu(crop, add_naive)
            x = crop.unsqueeze(0)
            x_pad, orig = pad_to_multiple(x, factor=8)
            with torch.autocast("cuda"):
                y = net(x_pad, torch.zeros(1, dtype=torch.long, device=dev))
            d, h, w = orig
            dose = (y[0, 0, :d, :h, :w].float() / DOSE_SCALE).cpu().numpy().astype(np.float32)
            key = (b.get("beam_idx", bi), ci)
            bb = tuple(int(v) for v in bbox)
            if on_frame is not None:   # stream: the writer starts while the GPU
                on_frame(key, dose, bb)   # moves on to the next control point
            else:
                out[key] = (dose, bb)
    return out
