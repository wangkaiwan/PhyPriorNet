"""Resumable fp16 training loop with EMA + OneCycleLR for the dose U-Net.
Optional wandb logging + in-loop validation, both fully guarded so a failure in
either NEVER interrupts training (important for unattended overnight runs)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

from doserad.data.dataset import _CH_SCALE
from doserad.losses.dose_loss import gamma_proxy, weighted_l1


def _init_wandb(cfg, run_dir):
    """Return a wandb run or None. Never raises. Defaults to offline mode (no auth
    needed) unless WANDB_MODE/api key say otherwise — safe for unattended runs."""
    if not cfg.get("wandb", False):
        return None
    try:
        import os
        import wandb
        # Default to online (user is logged in via netrc); set WANDB_MODE=offline
        # explicitly for unattended runs without auth. cfg can force it too.
        os.environ.setdefault("WANDB_MODE", cfg.get("wandb_mode", "online"))
        return wandb.init(project=cfg.get("wandb_project", "doserad2026"),
                          name=cfg["exp_name"], config=cfg,
                          dir=str(run_dir), resume="allow")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] disabled (init failed: {e})", flush=True)
        return None


def _wlog(run, data, step):
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] log failed: {e}", flush=True)


def build_optim(model, max_steps, lr=3e-4, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Ensure warmup is at least 2 steps to avoid ZeroDivisionError in OneCycleLR
    # when total_steps is small (e.g. 20 * 0.05 = 1.0 step → division by zero).
    warmup_steps = max(2, int(max_steps * 0.05))
    pct_start = warmup_steps / max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=max_steps,
                                                pct_start=pct_start)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return opt, sched, scaler, ema


@torch.no_grad()
def _ema_update(ema, model, decay=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm, alpha=1 - decay)


def _save(run_dir, model, opt, sched, scaler, ema, step, fname="state.pt"):
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (fname + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "ema": ema.state_dict(), "step": step,
                "rng": torch.get_rng_state()}, tmp)
    tmp.replace(run_dir / fname)


def train_steps(model, loader, opt, sched, scaler, ema, cfg, run_dir,
                max_steps, resume=False, device=None, init_from=None):
    run_dir = Path(run_dir)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev); ema.to(dev)
    step = 0
    if resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        ema.load_state_dict(st["ema"]); step = st["step"]
        torch.set_rng_state(st["rng"].cpu())
    elif init_from:
        # warm-start: load weights only (EMA preferred), fresh opt/sched/step.
        st = torch.load(init_from, map_location=dev)
        w = st.get("ema", st.get("model"))
        model.load_state_dict(w); ema.load_state_dict(w)
        print(f"warm-started weights from {init_from}", flush=True)

    wrun = _init_wandb(cfg, run_dir)
    val_crops = []
    train_crops = []          # fixed train-crop subset for the train/val MAE gap (overfit signal)
    val_every = int(cfg.get("val_every", 0) or 0)
    snap_every = int(cfg.get("snap_every", 0) or 0)   # 0 = off, existing behaviour
    if val_every > 0:
        try:
            from doserad.eval.quickval import collect_val_crops
            nvc = int(cfg.get("val_crops", 24))
            val_crops = collect_val_crops(cfg, nvc)
            train_crops = collect_val_crops(cfg, nvc, split="train")
            print(f"[val] {len(val_crops)} val + {len(train_crops)} train in-loop crops", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[val] in-loop validation disabled ({e})", flush=True)

    losses = []
    best_mae = float("inf")   # track best (lowest) in-loop val masked-MAE -> keep best.pt
                              # (smoother than the noisy per-CP gamma; fixes PITFALLS 12b)
    model.train()
    use_amp = (dev != "cpu") and torch.cuda.is_available()
    amp_device = dev.split(":")[0] if isinstance(dev, str) else "cuda"
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            x = batch["input"].to(dev); gt = batch["dose"].to(dev)
            m = batch["modality"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device, enabled=use_amp):
                pred = model(x, m)
                het_w = cfg.get("het_w", 0.0); lung_w = cfg.get("lung_w", 0.0)
                dens = x[:, 0:1] * float(_CH_SCALE[0]) if (het_w or lung_w) else None
                loss = weighted_l1(pred, gt, cfg["hi_frac"], cfg["hi_w"],
                                   grad_w=cfg.get("grad_w", 0.0),
                                   het_w=het_w, lung_w=lung_w, density=dens,
                                   lung_lo=cfg.get("lung_lo", 0.05),
                                   lung_hi=cfg.get("lung_hi", 0.6))
                gw = cfg.get("gamma_w", 0.0)
                if gw > 0:
                    warm = cfg.get("gamma_warmup", 0)
                    gw_eff = gw * min(1.0, step / warm) if warm > 0 else gw
                    if gw_eff > 0:
                        loss = loss + gw_eff * gamma_proxy(
                            pred.float(), gt.float(),
                            spacing_mm=cfg.get("dose_spacing_mm", 2.0))
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            _ema_update(ema, model)
            step += 1
            losses.append(float(loss.detach().cpu()))
            log_every = int(cfg.get("log_every", 200))       # console + wandb cadence
            if step % log_every == 0:
                recent = sum(losses[-log_every:]) / len(losses[-log_every:])
                pmax = float(pred.detach().max())
                lr = sched.get_last_lr()[0]
                print(f"step {step}/{max_steps} | loss {recent:.4f} | "
                      f"pred_max {pmax:.3f} | lr {lr:.2e}", flush=True)
                _wlog(wrun, {"train/loss": recent, "train/pred_max": pmax,
                             "train/lr": lr}, step)
            if step % int(cfg["ckpt_every"]) == 0:            # checkpoint cadence (heavy I/O)
                _save(run_dir, model, opt, sched, scaler, ema, step)
            if snap_every and step % snap_every == 0:
                # EMA weights only (~68 MB vs ~273 MB with optimiser state) at a fixed cadence, so
                # the step count can be chosen AFTER training instead of guessed before it. The
                # all-75 runs kept only best.pt and state.pt, and their "best" was selected on 4
                # patients that were in the training set (splits_all75 fold_all overlaps by 4), so
                # there was never a signal that could have caught over-training -- and the evidence
                # says we ran into it (all75_p2_ftg at 80k scores 75.11 on a patient it trained on,
                # where a 5CV model that never saw it scores 95.51).
                torch.save({"ema": ema.state_dict(), "step": step},
                           run_dir / f"snap_{step:06d}.pt")
            if val_every and step % val_every == 0 and val_crops:
                try:
                    from doserad.eval.quickval import quick_validate, masked_mae_only
                    ema.eval()
                    vmae = masked_mae_only(ema, val_crops, dev)
                    tmae = masked_mae_only(ema, train_crops, dev) if train_crops else float("nan")
                    # gamma is the expensive part (CPU DTA search) — only if explicitly
                    # asked (val_gamma_crops>0). Default: MAE-only; full gamma at final_eval.
                    ng = int(cfg.get("val_gamma_crops", 0))
                    vg1 = vg3 = float("nan")
                    if ng > 0:
                        _, vg1, vg3 = quick_validate(ema, val_crops, dev, ng)
                    ema.train()
                    tag = ""
                    if vmae < best_mae:             # select best by LOWEST val MAE (smooth)
                        best_mae = vmae
                        _save(run_dir, model, opt, sched, scaler, ema, step,
                              fname="best.pt")
                        tag = " *best*"
                    gap = vmae - tmae               # >0 => val worse than train = overfit signal
                    logv = math.log10(vmae) if vmae > 0 else float("nan")
                    logt = math.log10(tmae) if tmae > 0 else float("nan")
                    gmsg = f" | g1 {vg1*100:.1f}%" if ng > 0 else ""
                    print(f"[val] step {step} | val MAE {vmae*100:.3f}% (logMAE {logv:.3f}) | "
                          f"train {tmae*100:.3f}% (logMAE {logt:.3f}) | gap {gap*100:+.3f}%{gmsg}{tag}",
                          flush=True)
                    log = {"val/masked_mae": vmae, "train/masked_mae": tmae,
                           "val/train_gap_mae": gap, "val/log_masked_mae": logv,
                           "train/log_masked_mae": logt, "val/best_masked_mae": best_mae}
                    if ng > 0:
                        log.update({"val/gamma_1pct_1mm": vg1, "val/gamma_3pct_3mm": vg3})
                    _wlog(wrun, log, step)
                except Exception as e:  # noqa: BLE001
                    print(f"[val] step {step} skipped ({e})", flush=True)
    _save(run_dir, model, opt, sched, scaler, ema, step)
    if wrun is not None:
        try:
            wrun.finish()
        except Exception:  # noqa: BLE001
            pass
    return losses
