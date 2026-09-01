"""Paper-grade 3-way per-CP comparison vs GT: our naive (Tier-1) | official
pyRadPlan-PB (AAA) | our v6 DL. For every CP that has a precomputed pyRadPlan dose
(`baseline_pyradplan/<pid>/<b>_<cp>.npz`), compute masked-MAE + local gamma 1%/1mm
(each method LSQ-scaled to GT over the high-dose region — fair SHAPE accuracy),
aggregate per-method overall, BY SITE (abdomen vs lung) and per-patient.

Run in the `doserad` env. Usage:
  conda run -n doserad python scripts/eval_baseline_3way.py --ckpt runs/v6_photon_ct_naive/best.pt
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np
import torch

from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.beam_metrics import masked_mae
from doserad.eval.gamma import gamma_pass
from doserad.eval.plan_predict import ROOT, predict_cp
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.physics.naive_dose import compute_naive_dose

PB_DIR = Path("/home/kaiwang/doserad2026_workdir/baseline_pyradplan")
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon")


def _place(crop, bbox, shape):
    out = np.zeros(shape, np.float32)
    z0, z1, y0, y1, x0, x1 = bbox
    out[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = crop
    return out


def _lsq_metrics(pred, gt, spacing):
    m = gt >= 0.1 * gt.max()
    if not m.any():
        return np.nan, np.nan
    k = float((pred[m] * gt[m]).sum() / ((pred[m] ** 2).sum() + 1e-12))
    p = pred * k
    return masked_mae(p, gt) * 100, gamma_pass(p, gt, spacing, float(gt.max()),
                                                dose_pct=1.0, dta_mm=1.0) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/kaiwang/doserad2026_workdir/runs/v6_photon_ct_naive/best.pt")
    ap.add_argument("--out", default="/home/kaiwang/doserad2026_workdir/runs/v6_photon_ct_naive/baseline_3way.csv")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--label", default="dl", help="name for the DL column in stdout")
    args = ap.parse_args()
    dev = args.device
    net = DoseUNet3D(in_ch=6, base=48, levels=4)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev)["ema"]); net.to(dev).eval()

    rows = []
    pb_files = sorted(PB_DIR.glob("*/*.npz"))
    print(f"{len(pb_files)} CPs with pyRadPlan dose", flush=True)
    ct_cache = {}
    for f in pb_files:
        pid = f.parent.name
        b_cp = f.stem                       # "<beam>_<cp:03d>"
        b, cp3 = b_cp.split("_")
        if pid not in ct_cache:
            ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
            ct_cache.clear(); ct_cache[pid] = (ct.array.shape, ct.spacing)
        shape, spacing = ct_cache[pid]
        # GT (full grid)
        gt = load_mha(Path(ROOT) / pid / "dose" / f"Dose_B{b}_CP{cp3}.mha").array.astype(np.float32)
        # pyRadPlan (reconstruct full from saved bbox crop)
        z = np.load(f); pb = _place(z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"]), shape)
        # our naive + DL from the cached channels
        cf = CACHE / pid / f"{b}_{cp3}.npz"
        if not cf.exists():
            continue
        cz = np.load(cf); ch = cz["channels"].astype(np.float32); cbb = tuple(int(v) for v in cz["bbox"])
        naive = _place(compute_naive_dose(ch), cbb, shape)
        dl = _place(predict_cp(net, normalize_channels(ch, add_naive=True).astype(np.float32), dev) / DOSE_SCALE,
                    cbb, shape)
        # crop everything to GT's high-dose neighborhood for fast gamma
        mm = gt > 0.05 * gt.max()
        if not mm.any():
            continue
        zz, yy, xx = np.where(mm); mgn = 4
        sl = (slice(max(zz.min() - mgn, 0), zz.max() + mgn + 1),
              slice(max(yy.min() - mgn, 0), yy.max() + mgn + 1),
              slice(max(xx.min() - mgn, 0), xx.max() + mgn + 1))
        gtc = gt[sl]
        site = "lung" if "THB" in pid else "abdomen"
        r = {"patient": pid, "site": site, "cp": b_cp}
        for name, arr in [("naive", naive[sl]), ("pyradplan", pb[sl]), ("dl", dl[sl])]:
            mae, g1 = _lsq_metrics(arr, gtc, spacing)
            r[f"{name}_mae"] = mae; r[f"{name}_g1"] = g1
        rows.append(r)
        print(f"  {pid} {b_cp} ({site}): MAE naive {r['naive_mae']:.1f} / AAA {r['pyradplan_mae']:.1f} / "
              f"DL {r['dl_mae']:.1f}  | g1 naive {r['naive_g1']:.0f} / AAA {r['pyradplan_g1']:.0f} / "
              f"DL {r['dl_g1']:.0f}", flush=True)

    def agg(sub, key):
        v = [r[key] for r in sub if not np.isnan(r[key])]
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"),) * 2

    for label, sub in [("ALL", rows), ("ABDOMEN", [r for r in rows if r["site"] == "abdomen"]),
                       ("LUNG", [r for r in rows if r["site"] == "lung"])]:
        if not sub:
            continue
        print(f"\n=== {label} (n={len(sub)} CPs) — masked-MAE% / gamma1%/1mm% ===")
        for m in ("naive", "pyradplan", "dl"):
            ma, sa = agg(sub, f"{m}_mae"); ga, gsa = agg(sub, f"{m}_g1")
            print(f"  {m:10}: MAE {ma:5.1f}±{sa:4.1f}   gamma {ga:5.1f}±{gsa:4.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
