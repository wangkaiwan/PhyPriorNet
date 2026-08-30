"""Train a proton-CT per-beamlet dose model (Task3). NO-PRIOR (in_ch4) or WITH-PRIOR (in_ch5,
+pyRadPlan PB prior) — the v13-style prior A/B. NEW file; reuses the v13 loop helpers
(_save resumable state.pt, _init_wandb/_wlog/build_optim/_ema_update) + weighted_l1. Proton dose
scale + val MAE are proton-specific (PROTON_DOSE_SCALE=1e3), NOT photon's quickval (which hardcodes
DOSE_SCALE 1e4 + 2mm spacing). Residual learning when pb_prior present.
"""
from __future__ import annotations
import argparse, json, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from doserad.data.proton_dataset import ProtonDoseDataset, build_proton_rows, PROTON_DOSE_SCALE
from doserad.model.unet3d import DoseUNet3D
from doserad.losses.dose_loss import weighted_l1, gamma_proxy
from doserad.train.loop import build_optim, _ema_update, _save, _init_wandb, _wlog


@torch.no_grad()
def _val_mae(ema, crops, dev):
    """Mean masked-MAE (fraction of Gy via PROTON_DOSE_SCALE) over fixed val crops."""
    amp = "cuda" if dev != "cpu" else "cpu"
    errs = []
    for ch, gt in crops:                                 # ch normalised input, gt Gy
        x = torch.as_tensor(ch).unsqueeze(0).to(dev)
        with torch.autocast(amp, enabled=(dev != "cpu")):
            y = ema(x, torch.zeros(1, dtype=torch.long, device=dev))
        pred = (y[0, 0].float() / PROTON_DOSE_SCALE).cpu().numpy()
        m = gt >= 0.1 * gt.max() if gt.max() > 0 else np.zeros_like(gt, bool)
        if m.any():
            errs.append(float(np.abs(pred[m] - gt[m]).mean()))
    return float(np.mean(errs)) if errs else float("nan")


def _collect_crops(rows, cache_dir, prior_dir, patch, n, seed, wepl_dir=None):
    ds = ProtonDoseDataset(rows, cache_dir, prior_dir=prior_dir, wepl_dir=wepl_dir, patch=patch, fg_prob=1.0, seed=seed)
    if len(ds) == 0:
        return []
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    out = []
    for j in idx:
        b = ds[int(j)]
        out.append((b["input"].astype(np.float32), (b["dose"][0] / PROTON_DOSE_SCALE).astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg["cache_dir"]; prior_dir = cfg.get("prior_dir"); wepl_dir = cfg.get("wepl_dir")
    patch = tuple(cfg["patch"])

    splits = json.load(open(cfg["splits"]))
    fold = splits[f"fold_{cfg['fold']}"]
    tr_rows = build_proton_rows(cache_dir, fold["train"])
    va_rows = build_proton_rows(cache_dir, fold["val"])
    print(f"[data] {len(tr_rows)} train + {len(va_rows)} val beamlets (cache {cache_dir}, prior {prior_dir})", flush=True)
    if not tr_rows:
        print("NO training beamlets cached yet — abort.", flush=True); return

    ds = ProtonDoseDataset(tr_rows, cache_dir, prior_dir=prior_dir, wepl_dir=wepl_dir, patch=patch, fg_prob=cfg["fg_prob"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True,
                        persistent_workers=cfg.get("num_workers", 8) > 0)

    nvc = int(cfg.get("val_crops", 240))
    val_crops = _collect_crops(va_rows, cache_dir, prior_dir, patch, nvc, 123, wepl_dir) if va_rows else []
    train_crops = _collect_crops(tr_rows, cache_dir, prior_dir, patch, nvc, 321, wepl_dir)
    print(f"[val] {len(val_crops)} val + {len(train_crops)} train in-loop crops", flush=True)

    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev)
    max_steps = a.max_steps or cfg["max_steps"]
    opt, sched, scaler, ema = build_optim(net, max_steps, lr=cfg["lr"], wd=cfg["weight_decay"])
    ema.to(dev)

    step = 0
    if a.resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", map_location=dev)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); ema.load_state_dict(st["ema"]); step = st["step"]
        torch.set_rng_state(st["rng"].cpu()); print(f"[resume] step {step}", flush=True)
    elif cfg.get("init_from"):
        # v13ft-style finetune: load WEIGHTS only (EMA preferred), fresh opt/sched/step + small LR.
        st = torch.load(cfg["init_from"], map_location=dev)
        w = st.get("ema", st.get("model"))
        net.load_state_dict(w); ema.load_state_dict(w)
        print(f"[init_from] warm-started weights from {cfg['init_from']} (fresh opt/sched, step 0)", flush=True)

    # OPT-IN knowledge distillation (fold experiment 2026-08-21): if teacher_ckpt + distill_w>0, add a
    # teacher-matching term. Default off (distill_w 0 / no teacher_ckpt) -> behavior UNCHANGED.
    teacher = None
    distill_w = float(cfg.get("distill_w", 0.0))
    if cfg.get("teacher_ckpt") and distill_w > 0:
        teacher = DoseUNet3D(in_ch=cfg["in_ch"], base=int(cfg.get("teacher_base", 48)),
                             levels=cfg["levels"], bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
        tst = torch.load(cfg["teacher_ckpt"], map_location=dev)
        teacher.load_state_dict(tst.get("ema", tst.get("model")))
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[distill] teacher(base{cfg.get('teacher_base',48)}) <- {cfg['teacher_ckpt']} | distill_w={distill_w}", flush=True)

    wrun = _init_wandb(cfg, run_dir)
    het_w, lung_w = cfg.get("het_w", 0.0), cfg.get("lung_w", 0.0)
    log_every = int(cfg.get("log_every", 200)); ckpt_every = int(cfg.get("ckpt_every", 5000))
    val_every = int(cfg.get("val_every", 5000))
    losses = []; best = float("inf"); net.train(); it = iter(loader)
    while step < max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        x = b["input"].to(dev); gt = b["dose"].to(dev); dens = b["density"].to(dev)
        m = b["modality"].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=(dev != "cpu")):
            pred = net(x, m)
            _lwargs = dict(grad_w=cfg.get("grad_w", 0.0), het_w=het_w, lung_w=lung_w,
                           density=dens if (het_w or lung_w) else None,
                           lung_lo=cfg.get("lung_lo", 0.02), lung_hi=cfg.get("lung_hi", 0.6))
            loss = weighted_l1(pred, gt, cfg["hi_frac"], cfg["hi_w"], **_lwargs)
            if teacher is not None:
                with torch.no_grad():
                    tpred = teacher(x, m)
                loss = loss + distill_w * weighted_l1(pred, tpred, cfg["hi_frac"], cfg["hi_w"], **_lwargs)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); _ema_update(ema, net)
        step += 1; losses.append(float(loss.detach().cpu()))

        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            pmax = float(pred.detach().max()); lr = sched.get_last_lr()[0]
            print(f"step {step}/{max_steps} | loss {recent:.4f} | pred_max {pmax:.3f} | lr {lr:.2e}", flush=True)
            _wlog(wrun, {"train/loss": recent, "train/pred_max": pmax, "train/lr": lr}, step)
        if step % ckpt_every == 0:
            _save(run_dir, net, opt, sched, scaler, ema, step)
        if val_every and step % val_every == 0 and val_crops:
            try:
                ema.eval()
                vmae = _val_mae(ema, val_crops, dev); tmae = _val_mae(ema, train_crops, dev) if train_crops else float("nan")
                ema.train(); tag = ""
                if vmae < best:
                    best = vmae; _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt"); tag = " *best*"
                gap = vmae - tmae
                print(f"[val] step {step} | val MAE {vmae*1e3:.3f}e-3 Gy | train {tmae*1e3:.3f}e-3 | gap {gap*1e3:+.3f}{tag}", flush=True)
                _wlog(wrun, {"val/masked_mae": vmae, "train/masked_mae": tmae, "val/train_gap_mae": gap,
                             "val/best_masked_mae": best}, step)
            except Exception as e:  # noqa: BLE001
                print(f"[val] step {step} skipped ({e})", flush=True)
    _save(run_dir, net, opt, sched, scaler, ema, step)
    if wrun is not None:
        try: wrun.finish()
        except Exception: pass
    print(f"DONE. best val MAE {best*1e3:.3f}e-3 Gy", flush=True)


if __name__ == "__main__":
    main()
