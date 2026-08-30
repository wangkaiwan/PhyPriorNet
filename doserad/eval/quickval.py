"""Fast in-loop validation: per-beam masked-MAE + real per-CP local gamma 1%/1mm
on a fixed subset of validation-patient crops. Shared by the train loop (for
wandb logging) and scripts/monitor_val.py. Cheap enough to call every val_every
steps (~6-10 s/crop for gamma on a small --n-gamma subset)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.beam_metrics import masked_mae
from doserad.eval.gamma import gamma_pass


def collect_val_crops(cfg, n: int, split: str = "val"):
    splits = json.load(open(cfg["splits"]))
    val = splits[f"fold_{cfg['fold']}"][split]
    cache = Path(cfg["cache_dir"])
    aaa_cache = Path(cfg["aaa_cache_dir"]) if cfg.get("aaa_cache_dir") else None
    add_naive = bool(cfg.get("add_naive", False))
    scatter = bool(cfg.get("naive_scatter", False))
    skin_gate = cfg.get("naive_skin_gate")
    files = []
    for p in val:
        d = cache / p
        if d.exists():
            files += sorted(d.glob("*.npz"))
    if not files:
        return []
    files = files[:: max(1, len(files) // n)][:n]      # deterministic spread
    crops = []
    for f in files:
        z = np.load(f)
        aaa = np.load(aaa_cache / f.parent.name / f.name)["aaa"] if aaa_cache else None
        crops.append((normalize_channels(z["channels"], add_naive=add_naive,
                                          scatter=scatter, aaa=aaa,
                                          naive_skin_gate=skin_gate).astype(np.float32),
                      z["dose"].astype(np.float32)))     # gt absolute Gy
    return crops


def _pad_to_multiple(x: torch.Tensor, factor: int = 8):
    orig = x.shape[-3:]
    pad = []
    for s in reversed(orig):
        pad.extend([0, (-s) % factor])
    return torch.nn.functional.pad(x, pad), orig


@torch.no_grad()
def quick_validate(net, crops, device, n_gamma: int = 6):
    """Returns (mean masked-MAE, mean per-CP gamma 1%/1mm, mean per-CP gamma 3%/3mm),
    all as fractions. net is used in eval mode by the caller; modality fixed to 0 (CT)."""
    maes, g1, g3 = [], [], []
    amp = device.split(":")[0] if isinstance(device, str) else "cuda"
    for i, (ch, gt) in enumerate(crops):
        x = torch.as_tensor(ch).unsqueeze(0).to(device)
        xp, orig = _pad_to_multiple(x)
        with torch.autocast(amp, enabled=(str(device) != "cpu")):
            y = net(xp, torch.zeros(1, dtype=torch.long, device=device))
        d, h, w = orig
        pred = (y.squeeze(0).squeeze(0).float()[:d, :h, :w] / DOSE_SCALE).cpu().numpy()
        maes.append(masked_mae(pred, gt))
        if i < n_gamma:
            rx = float(gt.max())
            g1.append(gamma_pass(pred, gt, (2.0, 2.0, 2.0), rx=rx, dose_pct=1.0, dta_mm=1.0))
            g3.append(gamma_pass(pred, gt, (2.0, 2.0, 2.0), rx=rx, dose_pct=3.0, dta_mm=3.0))
    nanmean = lambda a: float(np.mean(a)) if a else float("nan")
    return nanmean(maes), nanmean(g1), nanmean(g3)


@torch.no_grad()
def masked_mae_only(net, crops, device):
    """Mean masked-MAE over crops, NO gamma (cheap) — for logging the train/val MAE
    gap (overfitting signal) on a fixed train-crop subset."""
    amp = device.split(":")[0] if isinstance(device, str) else "cuda"
    maes = []
    for ch, gt in crops:
        x = torch.as_tensor(ch).unsqueeze(0).to(device)
        xp, orig = _pad_to_multiple(x)
        with torch.autocast(amp, enabled=(str(device) != "cpu")):
            y = net(xp, torch.zeros(1, dtype=torch.long, device=device))
        d, h, w = orig
        pred = (y.squeeze(0).squeeze(0).float()[:d, :h, :w] / DOSE_SCALE).cpu().numpy()
        maes.append(masked_mae(pred, gt))
    return float(np.mean(maes)) if maes else float("nan")
