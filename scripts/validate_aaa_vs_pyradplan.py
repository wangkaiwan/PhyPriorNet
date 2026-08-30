"""EXACT numerical validation: our GPU `aaa_prior_dose` vs the official pyRadPlan
SVD-PB (AAA) engine, voxel-by-voxel on the SAME CP geometry + bbox.

For every precomputed pyRadPlan CP dose (`baseline_pyradplan/<pid>/<b>_<cp>.npz`),
recompute our AAA on the identical patient bbox, LSQ-scale ours to pyRadPlan over the
high-dose region (absolute MU calibration differs by design), and report agreement:
  - corr  (shape agreement; >0.97 ⇒ our reimplementation reproduces the engine)
  - rel-MAE (LSQ)  - mean |a·ours − pr| / mean(pr)
  - gamma 1%/1mm of ours vs pyRadPlan (treating pyRadPlan as reference)
Aggregated overall and by site (abdomen/lung). Run in env `doserad`.

  python scripts/validate_aaa_vs_pyradplan.py [--max-per-patient N] [--device cuda]
"""
import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.eval.gamma import gamma_pass
from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import load_photon_machine
from doserad.physics.priors.pencil_beam_aaa import load_aaa_kernel, aaa_prior_dose

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PB_DIR = Path("/home/kaiwang/doserad2026_workdir/baseline_pyradplan")


def metrics(ours, pr, spacing):
    m = pr >= 0.1 * pr.max()
    p = ours[m].astype(np.float64); g = pr[m].astype(np.float64)
    a = float((p * g).sum() / max((p * p).sum(), 1e-12))
    corr = float(np.corrcoef(p, g)[0, 1])
    rel_mae = float(np.abs(a * p - g).mean() / max(g.mean(), 1e-12))
    g1 = gamma_pass(ours * a, pr, spacing, float(pr.max()), dose_pct=1.0, dta_mm=1.0) * 100
    return corr, rel_mae * 100, g1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-patient", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    kernel = load_aaa_kernel()
    rows = []
    by_pat = defaultdict(list)
    for f in sorted(PB_DIR.glob("*/*.npz")):
        by_pat[f.parent.name].append(f)

    for pid, files in by_pat.items():
        if args.max_per_patient:
            files = files[:: max(1, len(files) // args.max_per_patient)][:args.max_per_patient]
        plan = load_photon_plan(f"{ROOT}/{pid}/{pid}.json")
        ct = load_mha(f"{ROOT}/{pid}/image/ct.mha")
        density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
        site = "lung" if "THB" in pid else "abdomen"
        for f in files:
            b, cp3 = f.stem.split("_")
            z = np.load(f)
            pr = z["dose"].astype(np.float32)
            bbox = tuple(int(v) for v in z["bbox"])
            beam = next(bb for bb in plan.beams if bb.beam_idx == int(b))
            cp = next(c for c in beam.control_points if c.cp_idx == int(cp3))
            iso = np.asarray(beam.iso_center, dtype=np.float64)
            src = beam_source_pos(iso, machine.sad_mm, cp.gantry_angle)
            ax, uh, vh = beam_basis(cp.gantry_angle)
            ours = aaa_prior_dose(density, ct.spacing, ct.origin, src, ax, uh, vh, iso,
                                  machine, np.asarray(cp.mlc_left_int_mm),
                                  np.asarray(cp.mlc_right_int_mm), kernel,
                                  out_bbox=bbox, device=args.device)
            if pr.max() <= 0:
                continue
            corr, rmae, g1 = metrics(ours, pr, ct.spacing)
            rows.append((pid, site, f.stem, corr, rmae, g1))
            print(f"  {pid} {f.stem} ({site}): corr {corr:+.3f}  rel-MAE {rmae:5.1f}%  "
                  f"gamma1/1 {g1:5.1f}%", flush=True)

    def agg(sub, i):
        v = [r[i] for r in sub if not np.isnan(r[i])]
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    for label, sub in [("ALL", rows),
                       ("ABDOMEN", [r for r in rows if r[1] == "abdomen"]),
                       ("LUNG", [r for r in rows if r[1] == "lung"])]:
        if not sub:
            continue
        c = agg(sub, 3); mm = agg(sub, 4); g = agg(sub, 5)
        print(f"\n=== {label} (n={len(sub)} CPs) ours vs pyRadPlan AAA ===")
        print(f"  corr      {c[0]:+.3f} ± {c[1]:.3f}")
        print(f"  rel-MAE   {mm[0]:5.2f} ± {mm[1]:.2f} %")
        print(f"  gamma1/1  {g[0]:5.1f} ± {g[1]:.1f} %")


if __name__ == "__main__":
    main()
