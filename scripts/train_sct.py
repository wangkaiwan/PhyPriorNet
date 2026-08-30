"""Train the in-domain sCT model on DoseRAD's paired CT/MRI.
Usage: conda run -n doserad python scripts/train_sct.py --config configs/experiments/v1_sct.yaml [--resume] [--max-steps N]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from doserad.data.sct_dataset import SCTPairedSliceDataset
from doserad.model.sct_unet import SCTUNet


def _save(run_dir, model, opt, scaler, ema, step):
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "state.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "ema": ema.state_dict(),
                "step": step, "rng": torch.get_rng_state()}, tmp)
    tmp.replace(run_dir / "state.pt")


@torch.no_grad()
def _ema_update(ema, model, decay=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm, alpha=1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    splits = json.load(open(cfg["splits"]))
    train_pids = splits[f"fold_{cfg['fold']}"]["train"]
    ds = SCTPairedSliceDataset(train_pids, cfg["root"],
                                slice_context=cfg["slice_context"],
                                patch_size=cfg.get("patch_size", 256))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SCTUNet(in_ch=2 * cfg["slice_context"] + 1,
                  base=cfg["base_ch"], levels=cfg["levels"]).to(device)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    max_steps = args.max_steps or cfg["max_steps"]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max_steps,
        pct_start=max(2, int(max_steps * 0.05)) / max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]
    step = 0
    if args.resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", map_location=device)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"]); ema.load_state_dict(st["ema"])
        step = st["step"]; torch.set_rng_state(st["rng"].cpu())
    net.train()
    use_amp = torch.cuda.is_available()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            x = batch["mri"].to(device); y = batch["ct"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                pred = net(x)
                loss = F.l1_loss(pred, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step(); _ema_update(ema, net); step += 1
            if step % cfg["ckpt_every"] == 0:
                print(f"step {step} l1 {float(loss):.4f}", flush=True)
                _save(run_dir, net, opt, scaler, ema, step)
    _save(run_dir, net, opt, scaler, ema, step)
    print(f"done -> {run_dir}/state.pt")


if __name__ == "__main__":
    main()
