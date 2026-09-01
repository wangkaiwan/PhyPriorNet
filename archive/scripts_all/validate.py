"""Run validation: for each val patient (whose crops are cached), predict each
CP, accumulate the plan, compute beam- and plan-level challenge metrics, write
CSVs. Photon-CT only (v1).

Usage: conda run -n doserad python scripts/validate.py \
           --config configs/experiments/v1_photon_ct.yaml --ckpt <state.pt>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from doserad.beam.parse import load_photon_plan
from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.beam_metrics import idd_rms, masked_mae
from doserad.eval.gamma import gamma_pass
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.plan_predict import (ROOT, predict_cp as _predict_cp,
                                       val_patients_with_cache as _val_patients_with_cache)
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.geometry import beam_basis


def validate_patient(net, pid, cfg, device, save_pred=None):
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    full_shape = ct.array.shape
    pred_cps = []; gt_cps = []
    beam_rows = []
    cache = Path(cfg["cache_dir"]) / pid
    for beam in plan.beams:
        # use middle CP's gantry as the beam-axis proxy for IDD
        mid_gantry = beam.control_points[len(beam.control_points) // 2].gantry_angle
        axis, _, _ = beam_basis(mid_gantry)
        for cp in beam.control_points:
            f = cache / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ch = normalize_channels(z["channels"],
                                    add_naive=cfg.get("add_naive", False)).astype(np.float32)
            pred_scaled = _predict_cp(net, ch, device)                # scaled-absolute
            # no GT leak: divide by the same constant used in training (the real
            # submission has no GT dose_max to rescale by).
            pred_abs = pred_scaled / DOSE_SCALE
            gt_abs = z["dose"].astype(np.float32)                     # absolute Gy
            bbox = tuple(int(v) for v in z["bbox"])
            pred_cps.append((pred_abs, bbox))
            gt_cps.append((gt_abs, bbox))
            beam_rows.append({"patient_id": pid,
                              "beam_idx": int(beam.beam_idx),
                              "cp_idx": int(cp.cp_idx),
                              "masked_mae": masked_mae(pred_abs, gt_abs),
                              "idd_rms": idd_rms(pred_abs, gt_abs, axis,
                                                  ct.spacing, ct.origin)})
    plan_pred = accumulate_plan(pred_cps, full_shape)
    plan_gt = accumulate_plan(gt_cps, full_shape)
    rx = float(plan_gt.max())                                          # proxy
    if save_pred:
        sp = Path(save_pred); sp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp / f"{pid}.npz", pred=plan_pred.astype(np.float32),
                            gt=plan_gt.astype(np.float32), ct=ct.array.astype(np.float32),
                            spacing=np.asarray(ct.spacing, np.float32), rx=np.float32(rx))
    s_mae = stratified_mae(plan_pred, plan_gt, rx)
    g1 = gamma_pass(plan_pred, plan_gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    g3 = gamma_pass(plan_pred, plan_gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    return beam_rows, {"patient_id": pid, "rx_proxy": rx, "strat_mae": s_mae,
                       "gamma_1pct_1mm": g1, "gamma_3pct_3mm": g3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-patients", type=int, default=None,
                    help="limit number of val patients (plan-gamma is slow ~tens of min each)")
    ap.add_argument("--only", default=None, help="comma-separated patient ids to run")
    ap.add_argument("--save-pred", default=None, help="dir to save plan pred/gt/ct npz")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain"))
    st = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(st["ema"])              # EMA weights
    net.to(device).eval()
    pids = _val_patients_with_cache(cfg)
    if args.only:
        want = set(args.only.split(",")); pids = [p for p in pids if p in want]
    if args.max_patients:
        pids = pids[:args.max_patients]
    print(f"val patients with cache: {len(pids)} -> {pids}")
    out = Path(args.out or (Path(cfg["run_root"]) / cfg["exp_name"] /
                            "val_metrics.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_rows = []
    beam_rows_all = []
    for pid in pids:
        br, pr = validate_patient(net, pid, cfg, device, save_pred=args.save_pred)
        beam_rows_all.extend(br); plan_rows.append(pr)
        print(f"{pid}: strat_mae {pr['strat_mae']:.4f} "
              f"gamma_1%/1mm {pr['gamma_1pct_1mm']:.3f} "
              f"gamma_3%/3mm {pr['gamma_3pct_3mm']:.3f}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "beam_idx", "cp_idx",
                                          "masked_mae", "idd_rms"])
        w.writeheader(); w.writerows(beam_rows_all)
    plan_out = out.with_name(out.stem + "_plans.csv")
    with open(plan_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "rx_proxy", "strat_mae",
                                          "gamma_1pct_1mm", "gamma_3pct_3mm"])
        w.writeheader(); w.writerows(plan_rows)
    print(f"wrote {out} and {plan_out}")


if __name__ == "__main__":
    main()
