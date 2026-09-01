"""Official pyRadPlan-AAA PLAN-LEVEL gamma on all val patients (the SCORED口径), to compare
against (a) our GPU AAA reimplementation (standalone plan g1 ~37.0) and (b) the DL models.
If official ≫ ours → our AAA reimpl has a bug.

Accumulates the precomputed official per-CP pyRadPlan dose (baseline_pyradplan/<pid>/<b>_<cp>.npz,
each with its own bbox) into the plan; GT plan from the channels cache; LSQ-scales the prior to GT
over the >=10% region; reports plan local gamma 1%/1mm by band, abdomen/lung split. Run in `doserad`.
  DOSERAD_SHARD=k/N python scripts/eval_pyradplan_plan.py
"""
import os, csv, statistics as st
from pathlib import Path
import numpy as np
from doserad.eval.gamma import gamma_pass, gamma_pass_by_band
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.plan_predict import ROOT, val_patients_with_cache
from doserad.io.mha import load_mha
import yaml

PB = Path("/home/kaiwang/doserad2026_workdir/baseline_pyradplan")
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon")


def patient(pid):
    ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha"); shape = ct.array.shape
    pp, gt = [], []
    for f in sorted((PB / pid).glob("*.npz")):
        cf = CACHE / pid / f.name
        if not cf.exists():
            continue
        d = np.load(f); z = np.load(cf)
        pp.append((d["dose"].astype(np.float32), tuple(int(v) for v in d["bbox"])))
        gt.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    plan_pp = accumulate_plan(pp, shape); plan_gt = accumulate_plan(gt, shape)
    m = plan_gt >= 0.1 * plan_gt.max()
    k = float((plan_pp[m] * plan_gt[m]).sum() / ((plan_pp[m] ** 2).sum() + 1e-12))
    plan_pp *= k; rx = float(plan_gt.max())
    g1 = gamma_pass_by_band(plan_pp, plan_gt, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    g3 = gamma_pass(plan_pp, plan_gt, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    return {"patient": pid, "site": "lung" if "THB" in pid else "abd",
            "g1": g1["overall"], "hi": g1["high"], "mid": g1["mid"], "lo": g1["low"],
            "g3": g3, "ncp": len(pp), "strat_mae": stratified_mae(plan_pp, plan_gt, rx)}


def main():
    cfg = yaml.safe_load(open("configs/experiments/v9_photon_ct_hardregions.yaml"))
    pids = val_patients_with_cache(cfg)
    k, n = (int(x) for x in os.environ.get("DOSERAD_SHARD", "0/1").split("/"))
    pids = pids[k::n]
    rows = []
    for pid in pids:
        r = patient(pid); rows.append(r)
        print(f"[official-AAA] {pid} ({r['site']}): plan g1 {r['g1']*100:.1f}% "
              f"(hi{r['hi']*100:.0f}/mid{r['mid']*100:.0f}/lo{r['lo']*100:.0f}) g3 {r['g3']*100:.1f}% "
              f"ncp {r['ncp']} stratMAE {r['strat_mae']*100:.2f}%", flush=True)
    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"official_aaa_plan_s{k}of{n}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if n == 1:
        for lab, sub in [("ALL", rows), ("ABD", [r for r in rows if r["site"] == "abd"]),
                         ("LUNG", [r for r in rows if r["site"] == "lung"])]:
            v = [r["g1"] * 100 for r in sub]
            print(f"=== official-AAA {lab}: plan g1 {st.mean(v):.1f}±{st.pstdev(v):.1f}% (n={len(sub)}) ===", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
