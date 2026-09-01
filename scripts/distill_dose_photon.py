"""Knowledge-distill a SMALLER/FASTER photon-CT dose U-Net (student, base=32/24) from the base=48
teacher (ftg_skinentry_photonct). Same PhotonCropDataset + weighted_l1 recipe; adds a teacher-matching
term so the student learns the teacher's smooth (denoised-vs-MC) dose in addition to the GT. Goal: 2x+
faster net at held gamma. New file — does NOT modify the read-only CT dose pipeline.

  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python scripts/distill_dose_photon.py \
      --config configs/experiments/cv/distill_photonct_b32_f0.yaml [--resume]
"""
from __future__ import annotations

import os
import argparse, json, math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from doserad.data.dataset import PhotonCropDataset, _CH_SCALE
from doserad.data.index import build_index
from doserad.model.unet3d import DoseUNet3D
from doserad.losses.dose_loss import weighted_l1
from doserad.train.loop import build_optim, _ema_update, _save, _init_wandb, _wlog

ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/photon/training")


def _load_teacher(ckpt, dev):
    t = DoseUNet3D(in_ch=6, base=48, levels=4, bottleneck="dilated").to(dev).eval()
    sd = torch.load(ckpt, map_location=dev); t.load_state_dict(sd.get("ema", sd.get("model")))
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); dev = "cuda"
    splits = json.load(open(cfg["splits"]))
    train_pids = set(splits[f"fold_{cfg['fold']}"]["train"])
    df = build_index(ROOT); rows = df[df.patient_id.isin(train_pids)].to_dict("records")
    ds = PhotonCropDataset(rows, cfg["cache_dir"], patch=tuple(cfg["patch"]),
                           fg_prob=cfg["fg_prob"], add_naive=bool(cfg.get("add_naive", False)),
                           augment=cfg.get("augment", False),
                           naive_skin_gate=cfg.get("naive_skin_gate"))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)

    student = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                         bottleneck=cfg.get("bottleneck", "dilated"))
    if cfg.get("init_student"):
        # warm-start pattern per PITFALLS "Continuing a finished run": NEW run dir,
        # EMA weights, fresh optimizer/scheduler/step counter
        ist = torch.load(cfg["init_student"], map_location="cpu")
        student.load_state_dict(ist.get("ema", ist.get("model", ist)))
        print(f"[init] student <- {cfg['init_student']} (EMA)", flush=True)
    distill_w = float(cfg.get("distill_w", 0.5))
    # distill_w == 0 -> pure GT supervision; skip teacher entirely (no load, no fwd)
    teacher = _load_teacher(cfg["teacher_ckpt"], dev) if distill_w > 0 else None
    max_steps = args.max_steps or cfg["max_steps"]
    opt, sched, scaler, ema = build_optim(student, max_steps, lr=cfg["lr"], wd=cfg["weight_decay"])
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    student.to(dev); ema.to(dev)

    step = 0
    if args.resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", map_location=dev)
        student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        ema.load_state_dict(st["ema"]); step = st["step"]; torch.set_rng_state(st["rng"].cpu())
        print(f"resumed at step {step}", flush=True)

    wrun = _init_wandb(cfg, run_dir)
    lw = lambda p, t, d: weighted_l1(p, t, cfg["hi_frac"], cfg["hi_w"], grad_w=cfg.get("grad_w", 0.0),
                                     het_w=cfg.get("het_w", 0.0), lung_w=cfg.get("lung_w", 0.0),
                                     density=d, lung_lo=cfg.get("lung_lo", 0.05), lung_hi=cfg.get("lung_hi", 0.6))
    student.train(); losses = []
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            x = batch["input"].to(dev); gt = batch["dose"].to(dev); m = batch["modality"].to(dev)
            opt.zero_grad(set_to_none=True)
            need_d = cfg.get("het_w", 0.0) or cfg.get("lung_w", 0.0)
            dens = x[:, 0:1] * float(_CH_SCALE[0]) if need_d else None
            with torch.autocast("cuda"):
                pred = student(x, m)
                loss_gt = lw(pred, gt, dens)
                if teacher is not None:
                    with torch.no_grad():
                        tpred = teacher(x, m)
                    loss_kd = lw(pred, tpred, dens)
                    loss = loss_gt + distill_w * loss_kd
                else:
                    loss_kd = torch.zeros((), device=dev)
                    loss = loss_gt
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            _ema_update(ema, student); step += 1
            losses.append(float(loss.detach().cpu()))
            le = int(cfg.get("log_every", 200))
            if step % le == 0:
                recent = sum(losses[-le:]) / len(losses[-le:])
                lr = sched.get_last_lr()[0]
                print(f"step {step}/{max_steps} | loss {recent:.4f} (gt {float(loss_gt):.4f} "
                      f"kd {float(loss_kd):.4f}) | pred_max {float(pred.max()):.3f} | lr {lr:.2e}", flush=True)
                _wlog(wrun, {"train/loss": recent, "train/loss_gt": float(loss_gt),
                             "train/loss_kd": float(loss_kd), "train/lr": lr}, step)
            if step % int(cfg["ckpt_every"]) == 0:
                _save(run_dir, student, opt, sched, scaler, ema, step)
    _save(run_dir, student, opt, sched, scaler, ema, step)
    if wrun is not None:
        try: wrun.finish()
        except Exception: pass
    print(f"done -> {run_dir}/state.pt")


if __name__ == "__main__":
    main()
