"""Route-A sCT evaluation (NO training): MRI --v1_sct--> sCT --v13ft(modality=ct)--> dose,
scored vs the GT dose on the 16 val patients. Measures the DOSE-level penalty of using sCT
instead of the real CT. Same plan-level local-gamma口径 as final_eval → directly comparable to
v13ft-on-real-CT = 95.71 (the ceiling). On-the-fly channels from sCT (no channel-cache regen).

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_sct_route.py [--max-patients N]
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
from doserad.data.sct_dataset import _normalize_mri
from doserad.eval.gamma import gamma_array, gamma_pass
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.plan_predict import ROOT, val_patients_with_cache
from doserad.eval.viz import render_sct_figure
from doserad.inference.pipeline import photon_inference
from doserad.io.mha import Volume, load_mha
from doserad.model.sct_unet import SCTUNet
from doserad.physics.machine import load_photon_machine

VIZ_DIR = Path("/home/kaiwang/doserad2026_workdir/runs/v13ft_photon_ct_dilated_hardregions/viz_sct")

SCT_CKPT = "/home/kaiwang/doserad2026_workdir/runs/v1_sct/state.pt"
DOSE_CKPT = "/home/kaiwang/doserad2026_workdir/runs/v13ft_photon_ct_dilated_hardregions/state.pt"
CFG = "configs/experiments/v13ft_photon_ct_dilated_hardregions.yaml"
SLICE_K = 2


def sct_infer(net, mr_arr, device):
    """MRI (z,y,x) -> sCT HU (z,y,x), slice-by-slice 2.5D (mirrors scripts/infer_pct)."""
    nz, ny, nx = mr_arr.shape
    mr_n = _normalize_mri(mr_arr)
    pad_h, pad_w = (-ny) % 8, (-nx) % 8
    if pad_h or pad_w:
        mr_n = np.pad(mr_n, ((0, 0), (0, pad_h), (0, pad_w)))
    out = np.zeros((nz, ny, nx), np.float32)
    for z in range(nz):
        stack = np.zeros((1, 2 * SLICE_K + 1, mr_n.shape[1], mr_n.shape[2]), np.float32)
        for j, zz in enumerate(range(z - SLICE_K, z + SLICE_K + 1)):
            if 0 <= zz < nz:
                stack[0, j] = mr_n[zz]
        x = torch.as_tensor(stack, device=device)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=(device == "cuda")):
                y = net(x).squeeze(0).squeeze(0).float().cpu().numpy()
        out[z] = y[:ny, :nx]
    return (out * 1000.0).astype(np.float32)


def gt_plan(pid, cfg, full_shape):
    """GT plan dose from the (CT-computed) per-CP cache crops (== raw Dose_*.mha, verified)."""
    gt_cps = []
    for f in sorted((Path(cfg["cache_dir"]) / pid).glob("*.npz")):
        z = np.load(f)
        gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
    return accumulate_plan(gt_cps, full_shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--label", default="sct_route")
    ap.add_argument("--sct-dir", default=None,
                    help="if set, load precomputed sCT from <dir>/<pid>/sCT.mha (e.g. VBoussot konfai output) "
                         "instead of running the v1_sct model")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    sct = None
    if not args.sct_dir:
        sct = SCTUNet(in_ch=2 * SLICE_K + 1, base=32, levels=4).to(dev).eval()
        sct.load_state_dict(torch.load(SCT_CKPT, map_location=dev)["ema"])

    pids = val_patients_with_cache(cfg)
    if args.max_patients:
        pids = pids[:args.max_patients]
    print(f"[sct-route] {len(pids)} val patients (sCT=v1_sct, dose=v13ft, modality=ct)", flush=True)

    rows = []
    for pid in pids:
        pdir = Path(ROOT) / pid
        plan = load_photon_plan(pdir / f"{pid}.json")
        mr = load_mha(pdir / "image" / "mr.mha")
        if args.sct_dir:
            sct_arr = load_mha(Path(args.sct_dir) / pid / "sCT.mha").array.astype(np.float32)
        else:
            sct_arr = sct_infer(sct, mr.array, dev)
        sct_vol = Volume(array=sct_arr, spacing=mr.spacing, origin=mr.origin, direction=mr.direction)
        out = photon_inference(sct_vol, plan, DOSE_CKPT, machine, modality="ct",
                               base_ch=cfg["base_ch"], levels=cfg["levels"], in_ch=cfg["in_ch"],
                               add_naive=cfg.get("add_naive", False),
                               bottleneck=cfg.get("bottleneck", "plain"))
        full_shape = sct_arr.shape
        plan_pred = np.zeros(full_shape, np.float32)
        for v in out.values():
            plan_pred += v
        del out
        plan_gt = gt_plan(pid, cfg, full_shape)
        rx = float(plan_gt.max())
        # N-a: crop to the dose region for FAST PyMedPhys gamma. margin 4 vox (8mm) >> DTA (1-3mm)
        # so every >=10% eval voxel + its DTA search stays inside the crop -> IDENTICAL result.
        zz, yy, xx = np.where(plan_gt >= 0.05 * rx)
        mgn = 4
        sl = (slice(max(int(zz.min()) - mgn, 0), int(zz.max()) + mgn + 1),
              slice(max(int(yy.min()) - mgn, 0), int(yy.max()) + mgn + 1),
              slice(max(int(xx.min()) - mgn, 0), int(xx.max()) + mgn + 1))
        pc, gc = plan_pred[sl], plan_gt[sl]
        g1c, g1m = gamma_array(pc, gc, mr.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        rate = lambda m: float((g1c[m] <= 1.0).mean()) if m.any() else float("nan")
        ov = rate(g1m)
        hi = rate(g1m & (gc >= 0.8 * rx))
        mid = rate(g1m & (gc >= 0.3 * rx) & (gc < 0.8 * rx))
        lo = rate(g1m & (gc >= 0.1 * rx) & (gc < 0.3 * rx))
        g3 = gamma_pass(pc, gc, mr.spacing, rx, dose_pct=3.0, dta_mm=3.0)
        site = "lung" if "THB" in pid else "abdomen"
        r = {"patient": pid, "site": site, "plan_g1": ov, "hi": hi, "mid": mid, "lo": lo,
             "plan_g3": g3, "strat_mae": stratified_mae(plan_pred, plan_gt, rx)}
        rows.append(r)
        print(f"  {pid} ({site}): sCT-dose PLAN g1 {ov*100:.1f}% "
              f"(hi{hi*100:.0f}/mid{mid*100:.0f}/lo{lo*100:.0f}) g3 {g3*100:.1f}% "
              f"stratMAE {r['strat_mae']*100:.2f}%", flush=True)
        # 5-panel viz: embed cropped gamma back into the full grid for anatomical slicing
        g1_full = np.full(full_shape, np.inf, np.float64); g1_full[sl] = g1c
        mask_full = np.zeros(full_shape, bool); mask_full[sl] = g1m
        try:
            VIZ_DIR.mkdir(parents=True, exist_ok=True)
            ct_arr = load_mha(pdir / "image" / "ct.mha").array
            render_sct_figure(patient=pid, mri=mr.array, sct=sct_arr, ct=ct_arr,
                              pred=plan_pred, gt=plan_gt, g1=g1_full, mask=mask_full, rx=rx,
                              out=str(VIZ_DIR / f"viz_sct_{pid}.png"))
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] {pid} skip ({e})", flush=True)

    def agg(sub, k):
        v = [r[k] for r in sub if not np.isnan(r[k])]
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    for lab, sub in [("ALL", rows), ("ABDOMEN", [r for r in rows if r["site"] == "abdomen"]),
                     ("LUNG", [r for r in rows if r["site"] == "lung"])]:
        if not sub:
            continue
        m, s = agg(sub, "plan_g1"); m3, s3 = agg(sub, "plan_g3")
        print(f"=== sCT-route {lab} (n={len(sub)}): plan g1 {m*100:.2f}±{s*100:.2f}% "
              f"| g3 {m3*100:.2f}±{s3*100:.2f}% ===  (ceiling: v13ft real-CT = 95.71)")
    out_csv = Path("/home/kaiwang/doserad2026_workdir/runs") / f"sct_route_{args.label}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
