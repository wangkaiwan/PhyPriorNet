"""Plan-level analysis aligned to the SCORED challenge metrics, with a focus on
WHERE the north-star metric (local gamma 1%/1mm) fails.

For each val patient (whose CP crops are cached) it accumulates the per-CP
predictions into the plan, then reports the locally-computable official metrics:
  - plan local gamma 1%/1mm (overall + by dose band: high ≥80%, mid 30-80%, low 10-30% of Rx)
  - plan local gamma 3%/3mm (trend proxy, not scored)
  - stratified plan MAE (3 bands, ÷ Rx)
  - mean beam masked-MAE and IDD-RMS over the plan's CPs
Aggregates mean±std across patients. Optionally compares TWO checkpoints
(e.g. v5 best.pt vs v6 best.pt) side by side.

DVH clinical score is NOT computed: the public training data has no OAR/target
contours (organizers score DVH on the hidden test set with their own structures).

Rx is the plan-GT max PROXY (no prescription dose in the data) — consistent for
A/B comparison, slightly off the true normalization.

Usage:
  conda run -n doserad python scripts/analyze_plan.py --config configs/experiments/v6_photon_ct_naive.yaml \
      --ckpt runs/v6_photon_ct_naive/best.pt [--ckpt-b runs/v5_photon_ct_fixed/best.pt --label-b v5] \
      [--max-patients 3]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np
import torch
import yaml

from doserad.beam.parse import load_photon_plan
from doserad.eval.beam_metrics import idd_rms, masked_mae
from doserad.eval.gamma import gamma_pass, gamma_pass_by_band
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.geometry import beam_basis

from doserad.eval.plan_predict import (ROOT, predict_cp,
                                       val_patients_with_cache)
from doserad.data.dataset import DOSE_SCALE, normalize_channels


def _load_net(ckpt, cfg, device):
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"], levels=cfg["levels"])
    st_ = torch.load(ckpt, map_location=device)
    net.load_state_dict(st_["ema"])
    return net.to(device).eval()


def analyze_patient(net, pid, cfg, device):
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    full_shape = ct.array.shape
    cache = Path(cfg["cache_dir"]) / pid
    pred_cps = []; gt_cps = []; beam_mae = []; beam_idd = []
    for beam in plan.beams:
        mid = beam.control_points[len(beam.control_points) // 2].gantry_angle
        axis, _, _ = beam_basis(mid)
        for cp in beam.control_points:
            f = cache / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ch = normalize_channels(z["channels"],
                                    add_naive=cfg.get("add_naive", False)).astype(np.float32)
            pred_abs = predict_cp(net, ch, device) / DOSE_SCALE
            gt_abs = z["dose"].astype(np.float32)
            bbox = tuple(int(v) for v in z["bbox"])
            pred_cps.append((pred_abs, bbox)); gt_cps.append((gt_abs, bbox))
            beam_mae.append(masked_mae(pred_abs, gt_abs))
            beam_idd.append(idd_rms(pred_abs, gt_abs, axis, ct.spacing, ct.origin))
    plan_pred = accumulate_plan(pred_cps, full_shape)
    plan_gt = accumulate_plan(gt_cps, full_shape)
    rx = float(plan_gt.max())                                   # proxy
    g1 = gamma_pass_by_band(plan_pred, plan_gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    g3 = gamma_pass(plan_pred, plan_gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    return {"patient_id": pid, "rx_proxy": rx,
            "strat_mae": stratified_mae(plan_pred, plan_gt, rx),
            "g1_overall": g1["overall"], "g1_high": g1["high"],
            "g1_mid": g1["mid"], "g1_low": g1["low"], "g3_overall": g3,
            "beam_mae": float(np.mean(beam_mae)) if beam_mae else float("nan"),
            "beam_idd": float(np.mean(beam_idd)) if beam_idd else float("nan")}


def _agg(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    if not vals:
        return float("nan"), float("nan")
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def _run_ckpt(ckpt, cfg, pids, device, label):
    net = _load_net(ckpt, cfg, device)
    rows = []
    for pid in pids:
        r = analyze_patient(net, pid, cfg, device)
        rows.append(r)
        print(f"[{label}] {pid}: g1 {r['g1_overall']*100:.1f}% "
              f"(hi {r['g1_high']*100:.0f}/mid {r['g1_mid']*100:.0f}/lo {r['g1_low']*100:.0f}) "
              f"g3 {r['g3_overall']*100:.1f}% strat_mae {r['strat_mae']*100:.2f}%", flush=True)
    return rows


def _summary(label, rows):
    print(f"\n=== {label} (mean±std over {len(rows)} patients) ===")
    for key, name in [("g1_overall", "plan gamma 1%/1mm  "),
                      ("g1_high", "  band high(≥80%)  "),
                      ("g1_mid", "  band mid(30-80%) "),
                      ("g1_low", "  band low(10-30%) "),
                      ("g3_overall", "plan gamma 3%/3mm  "),
                      ("strat_mae", "stratified plan MAE"),
                      ("beam_mae", "beam masked-MAE    "),
                      ("beam_idd", "beam IDD-RMS       ")]:
        m, s = _agg(rows, key)
        unit = "%" if key != "rx_proxy" else ""
        print(f"  {name}: {m*100:6.2f} ± {s*100:.2f} {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--config-b", default=None,
                    help="config for ckpt-b if it differs (e.g. v5 in_ch=5 vs v6 in_ch=6); "
                         "defaults to --config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-b", default=None, help="optional 2nd ckpt to compare")
    ap.add_argument("--label", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="plan gamma is slow (~tens of min/patient); limit for a quick read")
    ap.add_argument("--patients", default=None,
                    help="comma-separated specific patient ids to run (overrides max-patients)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pids = val_patients_with_cache(cfg)
    if args.patients:
        want = set(args.patients.split(","))
        pids = [p for p in pids if p in want]
    elif args.max_patients:
        pids = pids[:args.max_patients]
    print(f"analyzing {len(pids)} val patients: {pids}", flush=True)

    rows_a = _run_ckpt(args.ckpt, cfg, pids, device, args.label)
    _summary(args.label, rows_a)
    rows_b = None
    if args.ckpt_b:
        cfg_b = yaml.safe_load(open(args.config_b)) if args.config_b else cfg
        rows_b = _run_ckpt(args.ckpt_b, cfg_b, pids, device, args.label_b)
        _summary(args.label_b, rows_b)

    out = Path(args.out or (Path(cfg["run_root"]) / cfg["exp_name"] / "plan_analysis.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "patient_id", "g1_overall", "g1_high", "g1_mid", "g1_low",
              "g3_overall", "strat_mae", "beam_mae", "beam_idd", "rx_proxy"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_a:
            w.writerow({"label": args.label, **{k: r[k] for k in fields[1:]}})
        if rows_b:
            for r in rows_b:
                w.writerow({"label": args.label_b, **{k: r[k] for k in fields[1:]}})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
