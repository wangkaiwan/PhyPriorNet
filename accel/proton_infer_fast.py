"""Fast ray-centric proton inference: build_ray (3.24x exact) -> compiled dose net (2.65x) ->
cutoff-zero. Produces per-beamlet dose. This is the container's proton inner loop.

infer_plan_fast(image, plan_json_beams, density, net_compiled, machine, cutoff_map) yields
per-beamlet (key -> dose_np). Groups beamlets by ray; one WEPL/geometry compute per ray.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from doserad.data.proton_dataset import PROTON_DOSE_SCALE
from accel.proton_build_ray import build_ray

_r16 = lambda v: -(-v // 16) * 16


@torch.no_grad()
def infer_beams(image, beams, density, net, machine, device="cuda", cutoff=0.0):
    """beams: plan['beams'] list (json). density: gpu tensor. net: (compiled) DoseUNet3D.
    Returns {(beam_idx,ray_idx,l): dose_np}. cutoff: minimum_cutoff (dose<=cutoff -> 0)."""
    dev = device
    out = {}
    for b in beams:
        for r in b["rays"]:
            bls = [dict(energy=bl["energy"], bbox=bl["bbox"], key=(b["beam_idx"], r["ray_idx"], l))
                   for l, bl in enumerate(r["beamlets"]) if "bbox" in bl]
            if not bls:
                continue
            stacks = build_ray(image, r["ray_source"], r["ray_target"], bls,
                               machine=machine, density=density, device=dev)
            for (stack, bb), bl in zip(stacks, bls):
                d, h, w = stack.shape[-3:]
                x = F.pad(stack[None], (0, _r16(w) - w, 0, _r16(h) - h, 0, _r16(d) - d))
                with torch.autocast("cuda"):
                    y = net(x, torch.zeros(1, dtype=torch.long, device=dev))
                dose = (y[0, 0, :d, :h, :w].float() / PROTON_DOSE_SCALE)
                if cutoff > 0:
                    dose = torch.where(dose <= cutoff, torch.zeros_like(dose), dose)
                out[bl["key"]] = dose.cpu().numpy().astype(np.float32)
    return out
