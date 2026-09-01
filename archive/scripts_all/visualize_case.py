"""Qualitative GT-vs-prediction visualization for one photon case.

Panels per slice: CT (gray) with GT iso-lines, GT dose, our prediction, signed diff
(% of Rx), DTA map (mm, approx), and local gamma 1%/1mm map (fail = red). Saves a PNG
(MPLBACKEND=Agg — no display). For inspecting WHERE we fail (esp. lung).

  python scripts/visualize_case.py --config <cfg> --ckpt <run>/state.pt --patient 1THB121 \
      [--cp B_CP]   # default: whole plan (all CPs accumulated)  [--out fig.png] [--n-slices 3]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from doserad.beam.parse import load_photon_plan
from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.gamma import gamma_array
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.plan_predict import ROOT, predict_cp
from doserad.eval.viz import render_plan_figure
from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D


def _net(ckpt, cfg, dev):
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 5), base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain"), attn_heads=cfg.get("attn_heads", 4))
    net.load_state_dict(torch.load(ckpt, map_location=dev)["ema"]); return net.to(dev).eval()


def predict_full(net, pid, cfg, dev, only_cp=None):
    """Return (ct, gt_full, pred_full) on the full patient grid (absolute Gy)."""
    pdir = Path(ROOT) / pid
    plan = load_photon_plan(pdir / f"{pid}.json")
    ct = load_mha(pdir / "image" / "ct.mha")
    cache = Path(cfg["cache_dir"]) / pid
    aaa_cache = Path(cfg["aaa_cache_dir"]) / pid if cfg.get("aaa_cache_dir") else None
    add_naive = cfg.get("add_naive", False); scatter = cfg.get("naive_scatter", False)
    preds, gts = [], []
    for beam in plan.beams:
        for cp in beam.control_points:
            stem = f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
            if only_cp and stem != f"{only_cp}.npz":
                continue
            f = cache / stem
            if not f.exists():
                continue
            z = np.load(f)
            aaa = np.load(aaa_cache / stem)["aaa"] if aaa_cache else None
            ch = normalize_channels(z["channels"], add_naive=add_naive, scatter=scatter,
                                    aaa=aaa).astype(np.float32)
            bbox = tuple(int(v) for v in z["bbox"])
            preds.append((predict_cp(net, ch, dev) / DOSE_SCALE, bbox))
            gts.append((z["dose"].astype(np.float32), bbox))
    pred = accumulate_plan(preds, ct.array.shape)
    gt = accumulate_plan(gts, ct.array.shape)
    return ct, gt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--patient", required=True); ap.add_argument("--cp", default=None,
                    help="single CP 'B_CCC' (e.g. 0_045); default = whole plan")
    ap.add_argument("--views", default="axial,coronal,sagittal",
                    help="comma list of axial/coronal/sagittal (one slice each, through dose centroid)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cache", action="store_true", help="force recompute (ignore cached dose/gamma)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dta_search = 10.0
    # CACHE the heavy per-(patient,model) compute (plan dose + gamma maps) — only 16 val patients,
    # no point recomputing. Keyed by patient + exp_name + cp. ~hundreds of MB/patient on /data.
    cdir = Path("/data/kwang/doserad_viz_cache"); cdir.mkdir(parents=True, exist_ok=True)
    cf = cdir / f"{args.patient}_{cfg['exp_name']}_{args.cp or 'plan'}.npz"
    if cf.exists() and not args.no_cache:    # cache stores ct/gt/pred/g1/mask; DTA computed in render
        d = np.load(cf)
        ctarr = d["ct"]; sp = tuple(float(v) for v in d["sp"]); gt = d["gt"]; pred = d["pred"]
        g1 = d["g1"]; mask = d["mask"]; rx = float(d["rx"])
        print(f"loaded cache {cf} (rx={rx:.4g})", flush=True)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = _net(args.ckpt, cfg, dev)
        ct, gt, pred = predict_full(net, args.patient, cfg, dev, only_cp=args.cp)
        ctarr = ct.array; rx = float(gt.max()); sp = ct.spacing
        print(f"{args.patient}: rx(max GT)={rx:.4g} Gy; computing gamma maps...", flush=True)
        g1, mask = gamma_array(pred, gt, sp, rx, dose_pct=1.0, dta_mm=1.0)      # gamma 1%/1mm
        np.savez_compressed(cf, ct=ctarr, sp=np.array(sp), gt=gt, pred=pred, g1=g1, mask=mask, rx=rx)
        print(f"computed + cached {cf}", flush=True)
    out = args.out or f"/home/kaiwang/doserad2026_workdir/viz_{args.patient}_{args.cp or 'plan'}.png"
    render_plan_figure(patient=args.patient, ctarr=ctarr, sp=sp, gt=gt, pred=pred, g1=g1,
                       mask=mask, rx=rx, out=out, dta=None, dta_search=dta_search,
                       views=args.views, cp=args.cp)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
