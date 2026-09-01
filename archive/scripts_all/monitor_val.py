"""Live validation monitor for a running Photon-CT training — NON-DISRUPTIVE.

Polls `runs/<exp>/state.pt` (written atomically by the trainer every ckpt_every
steps). Whenever the step advances by >= --val-stride, loads the EMA weights and
computes fast, challenge-aligned metrics on a FIXED subset of validation-patient
crops, appends to `val_monitor.csv`, and prints one line per validation.

Metrics reported each cycle:
  - masked MAE per beam (challenge Level-1 metric: mean |pred-gt| over voxels
    >= 10% of beam-max GT, / beam-max), averaged over the crop subset, in %.
  - REAL PyMedPhys local gamma 1%/1mm pass rate, per-CP, averaged over a small
    --n-gamma subset (~6-10 s/crop). This is per-beam (single CP), a fast trend
    proxy for the challenge's plan-level (summed-CP) gamma.

The plan-level gamma, IDD RMS, and stratified plan MAE on the full summed dose are
computed separately by scripts/validate.py at milestones (slow, ~tens of min).

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python scripts/monitor_val.py \
      --config configs/experiments/v1_photon_ct.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.beam_metrics import masked_mae
from doserad.eval.gamma import gamma_pass
from doserad.model.unet3d import DoseUNet3D


def _pad_to_multiple(x: torch.Tensor, factor: int = 8):
    orig = x.shape[-3:]
    pad = []
    for s in reversed(orig):
        pad.extend([0, (-s) % factor])
    return torch.nn.functional.pad(x, pad), orig


def _collect_val_crops(cfg, n):
    splits = json.load(open(cfg["splits"]))
    val = splits[f"fold_{cfg['fold']}"]["val"]
    cache = Path(cfg["cache_dir"])
    files = []
    for p in val:
        d = cache / p
        if d.exists():
            files += sorted(d.glob("*.npz"))
    if not files:
        raise SystemExit("no cached val crops found")
    files = files[:: max(1, len(files) // n)][:n]      # deterministic spread
    crops = []
    for f in files:
        z = np.load(f)
        crops.append((normalize_channels(z["channels"]).astype(np.float32),
                      z["dose"].astype(np.float32)))    # gt absolute Gy
    return crops


def _validate(net, crops, device, n_gamma):
    # delegate to the shared helper (returns mae, gamma_1%/1mm, gamma_3%/3mm)
    from doserad.eval.quickval import quick_validate
    return quick_validate(net, crops, device, n_gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--interval", type=int, default=120, help="poll seconds")
    ap.add_argument("--n-crops", type=int, default=24, help="crops for masked-MAE")
    ap.add_argument("--n-gamma", type=int, default=6, help="crops for real gamma (slower)")
    ap.add_argument("--val-stride", type=int, default=2000, help="validate every N steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(cfg["run_root"]) / cfg["exp_name"]
    ckpt = run / "state.pt"
    crops = _collect_val_crops(cfg, args.n_crops)
    print(f"[monitor] {len(crops)} val crops; polling {ckpt} every {args.interval}s; "
          f"validate every {args.val_stride} steps", flush=True)
    out = run / "val_monitor.csv"
    if not out.exists():
        run.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(["step", "masked_mae", "gamma_1pct_1mm",
                                    "gamma_3pct_3mm", "time"])
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"], levels=cfg["levels"]).to(device).eval()
    last = -args.val_stride
    max_steps = int(cfg["max_steps"])
    while True:
        if ckpt.exists():
            try:
                st = torch.load(ckpt, map_location=device)
            except Exception:
                time.sleep(5)
                continue
            step = int(st["step"])
            if step - last >= args.val_stride or step >= max_steps:
                net.load_state_dict(st["ema"])
                mae, g1, g3 = _validate(net, crops, device, args.n_gamma)
                ts = time.strftime("%H:%M:%S")
                print(f"[val] step {step} | masked_MAE {mae * 100:.2f}% | "
                      f"gamma_1%/1mm {g1 * 100:.1f}% | gamma_3%/3mm {g3 * 100:.1f}% | {ts}",
                      flush=True)
                with open(out, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{mae:.5f}", f"{g1:.5f}",
                                            f"{g3:.5f}", ts])
                last = step
                if step >= max_steps:
                    print("[monitor] training reached max_steps; exiting", flush=True)
                    break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
