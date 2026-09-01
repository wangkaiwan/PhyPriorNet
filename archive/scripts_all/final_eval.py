"""End-of-training auto-evaluation over ALL 16 val patients, for a finished run.

PLAN-LEVEL ONLY (user 2026-06-13: "以后也只做 plan gamma") — same methodology as the
v5/v6/v8/v9 `analyze_plan.py` runs, so every model is directly comparable:
  - per-patient PLAN gamma 1%/1mm by dose band (high/mid/low) + 3%/3mm  (the SCORED metric)
  - per-patient stratified plan MAE
  - mean per-beam masked-MAE over the plan's CPs
NO per-CP gamma (it was ~1h/patient and v5/v6 never used it). Evaluates the chosen ckpt
(default last `state.pt`). Supports naive / scatter / AAA priors (reads from cfg) so the
input is identical to training. Shardable via --patients.

Chained after training:
  python scripts/final_eval.py --config <cfg> --ckpt <run>/state.pt --label <name>
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
from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.beam_metrics import masked_mae
from doserad.eval.gamma import gamma_pass, gamma_array
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.plan_predict import ROOT, predict_cp, val_patients_with_cache
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D


def _load_net(ckpt, cfg, device):
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain"), attn_heads=cfg.get("attn_heads", 4))
    net.load_state_dict(torch.load(ckpt, map_location=device)["ema"])
    return net.to(device).eval()


def eval_patient(net, pid, cfg, device, viz_dir=None, viz_label="final"):
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    full_shape = ct.array.shape
    cache = Path(cfg["cache_dir"]) / pid
    aaa_cache = Path(cfg["aaa_cache_dir"]) / pid if cfg.get("aaa_cache_dir") else None
    add_naive = cfg.get("add_naive", False)
    scatter = cfg.get("naive_scatter", False)
    pred_cps, gt_cps, beam_mae = [], [], []
    for beam in plan.beams:
        for cp in beam.control_points:
            stem = f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            f = cache / stem
            if not f.exists():
                continue
            z = np.load(f)
            aaa = np.load(aaa_cache / stem)["aaa"] if aaa_cache else None
            ch = normalize_channels(z["channels"], add_naive=add_naive,
                                    scatter=scatter, aaa=aaa).astype(np.float32)
            pred = predict_cp(net, ch, device) / DOSE_SCALE
            gt = z["dose"].astype(np.float32)
            bbox = tuple(int(v) for v in z["bbox"])
            pred_cps.append((pred, bbox)); gt_cps.append((gt, bbox))
            beam_mae.append(masked_mae(pred, gt))
    plan_pred = accumulate_plan(pred_cps, full_shape)
    plan_gt = accumulate_plan(gt_cps, full_shape)
    rx = float(plan_gt.max())
    # ONE 1%/1mm gamma array → derive band pass rates AND cache for viz (no extra gamma)
    g1a, g1m = gamma_array(plan_pred, plan_gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    rate = lambda m: float((g1a[m] <= 1.0).mean()) if m.any() else float("nan")
    ov, hi = rate(g1m), rate(g1m & (plan_gt >= 0.8 * rx))
    mid = rate(g1m & (plan_gt >= 0.3 * rx) & (plan_gt < 0.8 * rx))
    lo = rate(g1m & (plan_gt >= 0.1 * rx) & (plan_gt < 0.3 * rx))
    g3 = gamma_pass(plan_pred, plan_gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    if viz_dir is not None:                # per-patient PNG, inline (reuses arrays+g1; +1 DTA gamma pass, ~3-4s)
        try:
            from doserad.eval.viz import render_plan_figure
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
            png = Path(viz_dir) / f"viz_{pid}_{viz_label}.png"
            gp = render_plan_figure(patient=pid, ctarr=ct.array, sp=ct.spacing, gt=plan_gt,
                                    pred=plan_pred, g1=g1a, mask=g1m, rx=rx, out=str(png))
            print(f"  [viz] {pid} -> {png} (γ1/1 {gp:.1f}%)", flush=True)
        except Exception as e:  # noqa: BLE001  (never let viz break the eval)
            print(f"  [viz] {pid} skip ({e})", flush=True)
    return {"patient": pid, "site": "lung" if "THB" in pid else "abdomen",
            "plan_g1": ov, "plan_g1_hi": hi, "plan_g1_mid": mid, "plan_g1_lo": lo, "plan_g3": g3,
            "strat_mae": stratified_mae(plan_pred, plan_gt, rx),
            "beam_mae": float(np.mean(beam_mae)) if beam_mae else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="final")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--patients", default=None,
                    help="comma-separated pids to restrict to (for sharding across processes)")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip per-patient PNG visualization (on by default, ~3-4s/patient)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _load_net(args.ckpt, cfg, device)
    pids = val_patients_with_cache(cfg)
    if args.patients:
        want = set(args.patients.split(","))
        pids = [p for p in pids if p in want]
    print(f"[{args.label}] FINAL EVAL (plan-level) on {len(pids)} val patients: {pids}", flush=True)

    out = Path(args.out_dir or (Path(cfg["run_root"]) / cfg["exp_name"]))
    viz_dir = None if args.no_viz else str(out / "viz")

    pat_rows = []
    for pid in pids:
        r = eval_patient(net, pid, cfg, device, viz_dir=viz_dir, viz_label=args.label)
        pat_rows.append(r)
        print(f"  {pid} ({r['site']}): PLAN g1 {r['plan_g1']*100:.1f}% "
              f"(hi {r['plan_g1_hi']*100:.0f}/mid {r['plan_g1_mid']*100:.0f}/lo {r['plan_g1_lo']*100:.0f}) "
              f"g3 {r['plan_g3']*100:.1f}%  beam-MAE {r['beam_mae']*100:.2f}%  "
              f"strat-MAE {r['strat_mae']*100:.2f}%", flush=True)

    def agg(rows, k):
        v = [r[k] for r in rows if not np.isnan(r[k])]
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    for label, sub in [("ALL", pat_rows),
                       ("ABDOMEN", [r for r in pat_rows if r["site"] == "abdomen"]),
                       ("LUNG", [r for r in pat_rows if r["site"] == "lung"])]:
        if not sub:
            continue
        print(f"\n=== {args.label} / {label} (n={len(sub)} patients) ===")
        for k, n in [("plan_g1", "PLAN gamma1/1   "), ("plan_g1_hi", "  band hi      "),
                     ("plan_g1_mid", "  band mid     "), ("plan_g1_lo", "  band lo      "),
                     ("plan_g3", "PLAN gamma3/3   "), ("beam_mae", "mean beam MAE   "),
                     ("strat_mae", "stratified MAE  ")]:
            m, s = agg(sub, k)
            print(f"  {n}: {m*100:6.2f} ± {s*100:.2f} %")

    out.mkdir(parents=True, exist_ok=True)
    fields = ["patient", "site", "plan_g1", "plan_g1_hi", "plan_g1_mid", "plan_g1_lo",
              "plan_g3", "beam_mae", "strat_mae"]
    with open(out / f"final_per_patient_{args.label}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(pat_rows)
    print(f"\nwrote final_per_patient_{args.label}.csv to {out}")


if __name__ == "__main__":
    main()
