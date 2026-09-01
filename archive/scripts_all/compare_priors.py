"""Standalone PHYSICS-PRIOR comparison (no DL, no GPU training): for a given prior
scheme, accumulate the per-CP analytical prior into the plan, best-global-scale it
to the GT plan (LSQ over the >=10% region — local gamma is otherwise scale-blind),
and report plan-level metrics on ALL val patients (split abdomen/thorax).

This is the CHEAP Stage-A screen for the prior-upgrade comparison: rank candidate
priors by their standalone physics quality BEFORE spending DL training on the best
ones (Stage B = train v6-recipe with the winning prior). Measure BOTH this and the
final DL number — they can diverge (a prior may give good FEATURES without accurate
dose). See STATUS.md "STAGED ROADMAP".

Priors: `naive` (Tier-1 scatter-blind, current) ; `scatter` (v10-lite density-scaled,
when implemented) ; `pyradplan` (Tier-2, external — TODO #57).

Usage: conda run -n doserad python scripts/compare_priors.py \
           --config configs/experiments/v6_photon_ct_naive.yaml --prior naive [--max-patients N]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np
import yaml

from doserad.beam.parse import load_photon_plan
from doserad.eval.gamma import gamma_pass, gamma_pass_by_band
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.plan_predict import ROOT, val_patients_with_cache
from doserad.io.mha import load_mha
from doserad.physics.naive_dose import compute_naive_dose


AAA_CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_aaa")


def compute_prior(channels: np.ndarray, prior: str) -> np.ndarray:
    if prior == "naive":
        return compute_naive_dose(channels)
    if prior == "scatter":                       # v10-lite density-scaled scatter
        return compute_naive_dose(channels, scatter=True)
    # 'aaa' is loaded from the precomputed cache in analyze_patient (needs full geometry)
    raise ValueError(f"prior '{prior}' not implemented yet")


def analyze_patient(pid, cfg, prior):
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
    full_shape = ct.array.shape
    cache = Path(cfg["cache_dir"]) / pid
    aaa_cache = Path(cfg.get("aaa_cache_dir", AAA_CACHE)) / pid
    p_cps = []; gt_cps = []
    for f in sorted(cache.glob("*.npz")):
        z = np.load(f)
        if prior == "aaa":
            p = np.load(aaa_cache / f.name)["aaa"].astype(np.float32)   # Tier-2, precomputed
        else:
            p = compute_prior(z["channels"].astype(np.float32), prior)
        bbox = tuple(int(v) for v in z["bbox"])
        p_cps.append((p, bbox)); gt_cps.append((z["dose"].astype(np.float32), bbox))
    plan_p = accumulate_plan(p_cps, full_shape)
    plan_gt = accumulate_plan(gt_cps, full_shape)
    m = plan_gt >= 0.1 * plan_gt.max()                       # high-dose region
    k = float((plan_p[m] * plan_gt[m]).sum() / ((plan_p[m] ** 2).sum() + 1e-12))
    plan_p = plan_p * k                                       # best global scale (LSQ)
    rx = float(plan_gt.max())
    g1 = gamma_pass_by_band(plan_p, plan_gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    g3 = gamma_pass(plan_p, plan_gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    return {"patient_id": pid, "scale": k,
            "g1_overall": g1["overall"], "g1_high": g1["high"],
            "g1_mid": g1["mid"], "g1_low": g1["low"], "g3_overall": g3,
            "strat_mae": stratified_mae(plan_p, plan_gt, rx)}


def _agg(rows, key):
    v = [r[key] for r in rows if not np.isnan(r[key])]
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prior", default="naive")
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pids = val_patients_with_cache(cfg)
    if args.max_patients:
        pids = pids[:args.max_patients]
    print(f"PRIOR='{args.prior}' standalone on {len(pids)} val patients", flush=True)
    rows = []
    for pid in pids:
        r = analyze_patient(pid, cfg, args.prior)
        rows.append(r)
        site = "THX" if "THB" in pid else "ABD"
        print(f"[{args.prior}] {pid} ({site}): plan g1 {r['g1_overall']*100:5.1f}% "
              f"(hi {r['g1_high']*100:.0f}/mid {r['g1_mid']*100:.0f}/lo {r['g1_low']*100:.0f}) "
              f"g3 {r['g3_overall']*100:.1f}% strat_mae {r['strat_mae']*100:.2f}%", flush=True)
    abd = [r for r in rows if "THB" not in r["patient_id"]]
    thx = [r for r in rows if "THB" in r["patient_id"]]
    for label, sub in [("ALL", rows), ("ABDOMEN", abd), ("THORAX", thx)]:
        if not sub:
            continue
        m, s = _agg(sub, "g1_overall")
        ml, sl = _agg(sub, "g1_low")
        print(f"=== {args.prior} {label} (n={len(sub)}): plan g1 {m*100:.1f}±{s*100:.1f}% "
              f"| low-band {ml*100:.1f}±{sl*100:.1f}% ===", flush=True)
    out = Path(args.out or (Path(cfg["run_root"]) / cfg["exp_name"] /
                            f"prior_{args.prior}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
