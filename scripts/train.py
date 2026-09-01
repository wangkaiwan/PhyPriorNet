"""Train the Photon-CT dose U-Net from a YAML config.
Usage: conda run -n doserad python scripts/train.py --config configs/experiments/v1_photon_ct.yaml [--resume] [--max-steps N]
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import os

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from doserad.data.dataset import PhotonCropDataset
from doserad.data.index import build_index
from doserad.model.unet3d import DoseUNet3D
from doserad.train.loop import build_optim, train_steps

ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="warm-start model+EMA weights from a checkpoint (fresh opt/sched/step)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    splits = json.load(open(cfg["splits"]))
    train_pids = set(splits[f"fold_{cfg['fold']}"]["train"])
    df = build_index(ROOT)
    rows = df[df.patient_id.isin(train_pids)].to_dict("records")
    add_naive = bool(cfg.get("add_naive", False))
    ds = PhotonCropDataset(rows, cfg["cache_dir"], patch=tuple(cfg["patch"]),
                           fg_prob=cfg["fg_prob"], add_naive=add_naive,
                           augment=cfg.get("augment", False),
                           augment_rot=cfg.get("augment_rot", True),
                           scatter=cfg.get("naive_scatter", False),
                           aaa_cache_dir=cfg.get("aaa_cache_dir"),
                           naive_skin_gate=cfg.get("naive_skin_gate"))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 4),
                        pin_memory=True, drop_last=True)
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"],
                     levels=cfg["levels"], bottleneck=cfg.get("bottleneck", "plain"),
                     attn_heads=cfg.get("attn_heads", 4))
    max_steps = args.max_steps or cfg["max_steps"]
    opt, sched, scaler, ema = build_optim(net, max_steps, lr=cfg["lr"],
                                          wd=cfg["weight_decay"])
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]
    train_steps(net, loader, opt, sched, scaler, ema, cfg, run_dir,
                max_steps=max_steps, resume=args.resume,
                init_from=args.init_from or cfg.get("init_from"))
    print(f"done -> {run_dir}/state.pt")


if __name__ == "__main__":
    main()
