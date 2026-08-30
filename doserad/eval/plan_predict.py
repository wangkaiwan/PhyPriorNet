"""Shared per-CP prediction helpers for plan-level validation/analysis scripts.

Kept in the installed `doserad` package (not in `scripts/`) so both
scripts/validate.py and scripts/analyze_plan.py import them robustly regardless
of how the script is launched."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

# Raw dataset root (overridable via env; matches the workdir convention).
ROOT = os.environ.get("DOSERAD_PHOTON_ROOT",
                      "/data/kwang/DoseRad2026_raw/photon/training")


def pad_to_multiple(x: torch.Tensor, factor: int = 8):
    """Pad spatial dims (last 3) to the next multiple of `factor`. Returns the
    padded tensor and the original (d, h, w) for later unpadding."""
    orig = x.shape[-3:]
    pad = []
    for s in reversed(orig):
        pad.extend([0, (-s) % factor])
    return torch.nn.functional.pad(x, pad), orig


def predict_cp(net, ch_norm, device):
    """Predict one CP's scaled-absolute dose from normalized channels (C,d,h,w)."""
    x = torch.as_tensor(ch_norm).unsqueeze(0).to(device)
    m = torch.zeros(1, dtype=torch.long, device=device)
    x_pad, orig = pad_to_multiple(x, factor=8)
    with torch.no_grad():
        with torch.autocast(device_type=device.split(":")[0], enabled=(device != "cpu")):
            out = net(x_pad, m).squeeze(0).squeeze(0).float()
    d, h, w = orig
    return out[:d, :h, :w].cpu().numpy()


def val_patients_with_cache(cfg):
    """Val patient ids (for cfg's fold) that have cached CP crops."""
    splits = json.load(open(cfg["splits"]))
    val_pids = splits[f"fold_{cfg['fold']}"]["val"]
    cache = Path(cfg["cache_dir"])
    return [p for p in val_pids if (cache / p).exists()
            and any((cache / p).glob("*.npz"))]
