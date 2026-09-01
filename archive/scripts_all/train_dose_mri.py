"""Train an MRI->dose (exp1) or MRI+sCT->dose (exp2) model.

NEW file; imports the CT-photon pipeline READ-ONLY. The training loop, validation
metrics, wandb fields, OneCycle schedule, EMA, grad-clip, best-by-val-MAE selection
and the resumable `state.pt` checkpoint are all IDENTICAL to the v13 CT-dose loop
(doserad/train/loop.py) by reusing its helpers. The ONLY difference from v13 is the
model INPUT (MRI / MRI+sCT instead of CT physics channels) and that the het/lung
density for the loss comes from the dataset's GT-CT (batch["density"]) rather than
input channel 0.
"""
from __future__ import annotations
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from doserad.data.index import build_index
from doserad.data.mri_dose_dataset import MRIDoseDataset, DOSE_SCALE
from doserad.model.unet3d import DoseUNet3D
from doserad.losses.dose_loss import weighted_l1, gamma_proxy
from doserad.physics.machine import load_photon_machine
# v13 loop helpers (reused verbatim -> identical schedule / ckpt / wandb behaviour):
from doserad.train.loop import build_optim, _ema_update, _save, _init_wandb, _wlog
# v13 validation metric fns (quickval.masked_mae_only / quick_validate) are imported
# LAZILY inside the val block (like v13's loop.py) so a missing pymedphys only disables
# in-loop validation instead of crashing training startup.

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"


def collect_mri_crops(rows, cfg, hu_anchors, n, fg_prob=1.0, seed=123):
    """Deterministic (input_channels, gt_abs_Gy) crops for in-loop validation —
    same tuple format quickval.masked_mae_only / quick_validate consume for v13."""
    ds = MRIDoseDataset(rows, cfg["cache_dir"], ROOT, hu_anchors,
                        patch=tuple(cfg["patch"]), mode=cfg["mode"],
                        sct_dir=cfg.get("sct_dir"), sct_phys_dir=cfg.get("sct_phys_dir"), fg_prob=fg_prob, seed=seed)
    if len(ds) == 0:
        return []
    idx = np.linspace(0, len(ds) - 1, min(int(n), len(ds))).astype(int)
    crops = []
    for j in idx:
        b = ds[int(j)]
        ch = b["input"].astype(np.float32)                  # (C,z,y,x)
        gt = (b["dose"][0] / DOSE_SCALE).astype(np.float32)  # (z,y,x) absolute Gy
        crops.append((ch, gt))
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <run_dir>/state.pt (model/opt/sched/scaler/ema/step/rng)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(cfg["run_root"]) / cfg["exp_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    machine = load_photon_machine(MACHINE)

    splits = json.load(open(cfg["splits"]))
    fold = splits[f"fold_{cfg['fold']}"]
    df = build_index(ROOT)
    tr_rows = df[df.patient_id.isin(set(fold["train"]))].to_dict("records")
    va_rows = df[df.patient_id.isin(set(fold["val"]))].to_dict("records")

    ds = MRIDoseDataset(tr_rows, cfg["cache_dir"], ROOT, machine.hu_anchors,
                        patch=tuple(cfg["patch"]), mode=cfg["mode"],
                        sct_dir=cfg.get("sct_dir"), sct_phys_dir=cfg.get("sct_phys_dir"), fg_prob=cfg["fg_prob"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 4), pin_memory=True,
                        drop_last=True)

    # in-loop val + train crops (overfit gap) — same counts/metrics as v13
    nvc = int(cfg.get("val_crops", 240))
    val_crops = train_crops = []
    if int(cfg.get("val_every", 0) or 0) > 0:
        val_crops = collect_mri_crops(va_rows, cfg, machine.hu_anchors, nvc, seed=123)
        train_crops = collect_mri_crops(tr_rows, cfg, machine.hu_anchors, nvc, seed=321)
        print(f"[val] {len(val_crops)} val + {len(train_crops)} train in-loop crops", flush=True)

    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev)
    max_steps = a.max_steps or cfg["max_steps"]
    opt, sched, scaler, ema = build_optim(net, max_steps, lr=cfg["lr"],
                                          wd=cfg["weight_decay"])
    ema.to(dev)

    step = 0
    if a.resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", map_location=dev)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        ema.load_state_dict(st["ema"]); step = st["step"]
        torch.set_rng_state(st["rng"].cpu())
        print(f"[resume] from state.pt @ step {step}", flush=True)

    wrun = _init_wandb(cfg, run_dir)
    het_w, lung_w = cfg.get("het_w", 0.0), cfg.get("lung_w", 0.0)
    log_every = int(cfg.get("log_every", 200))
    ckpt_every = int(cfg.get("ckpt_every", 5000))
    val_every = int(cfg.get("val_every", 0) or 0)
    ng = int(cfg.get("val_gamma_crops", 0))
    use_amp = (dev != "cpu")
    losses = []
    best_mae = float("inf")
    net.train()
    it = iter(loader)
    while step < max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        x = b["input"].to(dev); gt = b["dose"].to(dev); dens = b["density"].to(dev)
        m = b["modality"].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            pred = net(x, m)
            loss = weighted_l1(pred, gt, cfg["hi_frac"], cfg["hi_w"],
                               grad_w=cfg.get("grad_w", 0.0),
                               het_w=het_w, lung_w=lung_w,
                               density=dens if (het_w or lung_w) else None,
                               lung_lo=cfg.get("lung_lo", 0.02),
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        _ema_update(ema, net)
        step += 1
        losses.append(float(loss.detach().cpu()))

        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            pmax = float(pred.detach().max())
            lr = sched.get_last_lr()[0]
            print(f"step {step}/{max_steps} | loss {recent:.4f} | "
                  f"pred_max {pmax:.3f} | lr {lr:.2e}", flush=True)
            _wlog(wrun, {"train/loss": recent, "train/pred_max": pmax,
                         "train/lr": lr}, step)
        if step % ckpt_every == 0:
            _save(run_dir, net, opt, sched, scaler, ema, step)
        if val_every and step % val_every == 0 and val_crops:
            try:
                from doserad.eval.quickval import masked_mae_only, quick_validate
                ema.eval()
                vmae = masked_mae_only(ema, val_crops, dev)
                tmae = masked_mae_only(ema, train_crops, dev) if train_crops else float("nan")
                vg1 = vg3 = float("nan")
                if ng > 0:
                    _, vg1, vg3 = quick_validate(ema, val_crops, dev, ng)
                ema.train()
                tag = ""
                if vmae < best_mae:
                    best_mae = vmae
                    _save(run_dir, net, opt, sched, scaler, ema, step, fname="best.pt")
                    tag = " *best*"
                gap = vmae - tmae
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
    _save(run_dir, net, opt, sched, scaler, ema, step)
    if wrun is not None:
        try:
            wrun.finish()
        except Exception:  # noqa: BLE001
            pass
    print(f"DONE. best val masked-MAE {best_mae:.5f}", flush=True)


if __name__ == "__main__":
    main()
