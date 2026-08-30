"""Plan-γ eval for the end-to-end dose-aware MRI->sCT->dose model.
Per val patient: MRI -> sCT -> density -> per-CP differentiable physics channels (no_grad) -> dose net,
accumulate ALL CPs into the plan, score plan-level local γ 1%/1mm + 3%/3mm vs the MC GT plan (same
method as eval_dose_mri / sCT route -> directly comparable to exp3 84.7 and the sCT route 82.6).
    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/eval_dose_e2e.py \
        --config configs/experiments/mri_dose_e2e.yaml --ckpt <run>/state.pt --label e2e_last
"""
from __future__ import annotations
import argparse, json, os, sys, csv, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.beam.parse import load_photon_plan
from doserad.physics.diff_channels import (hu_to_density_torch, radiological_depth_fast_torch,
                                           fluence_torch, naive_dose_torch)
from doserad.physics.diff_channels_skinentry import radiological_depth_skinentry_torch
from doserad.data.dataset import _CH_SCALE, _NAIVE_SCALE, DOSE_SCALE
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.gamma import gamma_array, gamma_pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_dose_e2e import E2E, CT_LO, CT_HI, _pad16, DENS_MAX


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--sw", action="store_true", help="generate sCT via sliding-window (128^3, "
                    "overlap 0.25, gaussian) instead of one full-volume forward — higher-quality sCT")
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--only", default=None); ap.add_argument("--save-pred", default=None)
    ap.add_argument("--sct-dir", default=None, help="use precomputed HU volume <dir>/<pid>.nii.gz as the "
                    "density source (e.g. classifier coarse CT) instead of the synth net -> dose from a "
                    "pure bulk-density assignment through v13")
    a = ap.parse_args()
    if a.sw:
        from monai.inferers import sliding_window_inference
    cfg = yaml.safe_load(open(a.config)); dev = "cuda"
    skin_entry = bool(cfg.get("naive_skin_gate", False))   # skin-entry depth + entered-gated naive prior (== CT path)
    rdepth_fn = radiological_depth_skinentry_torch if skin_entry else radiological_depth_fast_torch
    ROOT = cfg["root_dir"]; machine = load_photon_machine(f"{ROOT}/beam_parameters.json"); anchors = machine.hu_anchors
    cache = Path(cfg["cache_dir"]); mod = torch.zeros(1, dtype=torch.long, device=dev)
    net = E2E(cfg).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))
    img_ch = int(cfg["in_ch"]) > 6                   # B: dose net also takes raw MRI + sCT
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
    val = [p for p in val if (cache / p).is_dir()]
    if a.only: want = set(a.only.split(",")); val = [p for p in val if p in want]
    if a.max_patients: val = val[:a.max_patients]
    viz_dir = Path(cfg["run_root"]) / cfg["exp_name"] / "viz"; rows = []
    for pid in val:
        mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha")
        a_mr = mr.array.astype(np.float32); lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
        mr01 = torch.from_numpy(np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(dev)
        with torch.autocast("cuda"):
            if a.sct_dir:                                  # use a precomputed HU volume (e.g. coarse CT)
                cv = load_mha(Path(a.sct_dir) / f"{pid}.nii.gz").array.astype(np.float32)
                assert cv.shape == a_mr.shape, f"{pid}: sct {cv.shape} != mr {a_mr.shape}"
                sct01 = torch.from_numpy(np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)).to(dev)
            elif a.sw:
                sct01 = sliding_window_inference(mr01[None, None], (128, 128, 128), a.sw_batch,
                                                 net.synth, overlap=0.25, mode="gaussian")[0, 0]
            elif cfg.get("coarse_dir"):                    # classify-then-regress: 2-ch synth [MR, coarse]
                cv = load_mha(Path(cfg["coarse_dir"]) / f"{pid}.nii.gz").array.astype(np.float32)
                co = torch.from_numpy(np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)).to(dev)
                sct01 = net.sct01(torch.stack([mr01, co], 0)[None])[0, 0]
            else:
                sct01 = net.sct01(mr01[None, None])[0, 0]
            sct_hu = sct01 * (CT_HI - CT_LO) + CT_LO
            if cfg.get("density_direct"):
                density = (sct01.clamp(0, 1) * DENS_MAX).float()   # synth out -> density (clamped)
            else:
                density = hu_to_density_torch(sct_hu, anchors).float()
        ct_arr = load_mha(Path(ROOT) / pid / "image" / "ct.mha").array.astype(np.float32)
        body = ct_arr > -500
        hu_mae = float(np.abs(sct_hu.float().cpu().numpy() - ct_arr)[body].mean())
        pl = load_photon_plan(Path(ROOT) / pid / f"{pid}.json")
        geo = {}
        for b in pl.beams:
            for cp in b.control_points:
                iso = np.asarray(b.iso_center, np.float64)
                geo[f"{b.beam_idx}_{cp.cp_idx:03d}"] = (iso, beam_source_pos(iso, machine.sad_mm, cp.gantry_angle),
                                                        *beam_basis(cp.gantry_angle))
        full = a_mr.shape; pred_cps, gt_cps = [], []
        for f in sorted((cache / pid).glob("*.npz")):
            z = np.load(f); ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
            z0, z1, y0, y1, x0, x1 = bb; sl = (slice(z0, z1+1), slice(y0, y1+1), slice(x0, x1+1))
            iso, src, ax, u, v = geo[f.stem]
            with torch.autocast("cuda"):
                rdepth = rdepth_fn(density, mr.spacing, mr.origin, src, ax, u, v, iso, out_bbox=bb)
                cht = torch.from_numpy(ch).to(dev)
                fl = fluence_torch((cht[2] > 0).float(), rdepth)
                naive = naive_dose_torch(density[sl], rdepth, fl, cht[4], skin_gate=skin_entry)
                chans = [density[sl] / _CH_SCALE[0], rdepth / _CH_SCALE[1], fl / _CH_SCALE[2],
                         cht[3] / _CH_SCALE[3], cht[4] / _CH_SCALE[4], naive / _NAIVE_SCALE]
                if img_ch:
                    chans += [mr01[sl], sct01[sl]]
                inp = torch.stack(chans, 0)
                Z, Y, X = inp.shape[-3:]
                inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                pred = net.dose(inp, mod)[0, 0, :Z, :Y, :X].float() / DOSE_SCALE
            pred_cps.append((pred.cpu().numpy(), bb)); gt_cps.append((z["dose"].astype(np.float32), bb))
        bmaes = []
        for (pr_b, _), (gt_b, _) in zip(pred_cps, gt_cps):
            gm = gt_b.max()
            if gm <= 0: continue
            mk = gt_b >= 0.1 * gm
            if mk.any(): bmaes.append(float(np.abs(pr_b[mk] - gt_b[mk]).mean() / gm))
        beam_mae = float(np.mean(bmaes)) if bmaes else float("nan")
        pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
        rx = float(gt.max()); zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
        if a.save_pred:
            _sp = Path(a.save_pred); _sp.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_sp / f"{pid}.npz", pred=pp.astype(np.float32), gt=gt.astype(np.float32),
                                ct=ct_arr.astype(np.float32), spacing=np.asarray(mr.spacing, np.float32), rx=np.float32(rx))
        crop = (slice(max(int(zz.min())-m, 0), int(zz.max())+m+1), slice(max(int(yy.min())-m, 0), int(yy.max())+m+1),
                slice(max(int(xx.min())-m, 0), int(xx.max())+m+1))
        g1c, g1m = gamma_array(pp[crop], gt[crop], mr.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        ov = float((g1c[g1m] <= 1.0).mean()) if g1m.any() else float("nan")
        g2 = gamma_pass(pp[crop], gt[crop], mr.spacing, rx, dose_pct=2.0, dta_mm=2.0)
        g3 = gamma_pass(pp[crop], gt[crop], mr.spacing, rx, dose_pct=3.0, dta_mm=3.0)
        site = "lung" if "THB" in pid else "abdomen"
        rows.append({"patient": pid, "site": site, "plan_g1": ov, "plan_g2": g2, "plan_g3": g3, "beam_mae": beam_mae,
                     "strat_mae": stratified_mae(pp, gt, rx), "sct_hu_mae": hu_mae})
        print(f"  {pid} ({site}): PLAN g1 {ov*100:.1f}%  g2 {g2*100:.1f}%  g3 {g3*100:.1f}%  sCT HU-MAE {hu_mae:.1f}", flush=True)
        try:
            from doserad.eval.viz import render_plan_figure
            g1f = np.full(full, np.inf); g1f[crop] = g1c; mf = np.zeros(full, bool); mf[crop] = g1m
            viz_dir.mkdir(parents=True, exist_ok=True)
            render_plan_figure(patient=pid, ctarr=mr.array, sp=mr.spacing, gt=gt, pred=pp, g1=g1f, mask=mf,
                               rx=rx, out=str(viz_dir / f"viz_{pid}_{a.label}.png"))
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] {pid} skip ({e})", flush=True)
    out = Path(cfg["run_root"]).parent / "runs" / f"dose_mri_{a.label}.csv"
    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"dose_mri_{a.label}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    g = lambda s: st.mean([r["plan_g1"] for r in rows if s is None or r["site"] == s])
    hm = lambda s: st.mean([r["sct_hu_mae"] for r in rows if s is None or r["site"] == s])
    print(f"\n{a.label}{' [SW]' if a.sw else ''}: PLAN g1 ALL {g(None)*100:.1f} "
          f"(abd {g('abdomen')*100:.1f} / lung {g('lung')*100:.1f}) "
          f"| g2 ALL {st.mean([r['plan_g2'] for r in rows])*100:.1f} "
          f"| g3 ALL {st.mean([r['plan_g3'] for r in rows])*100:.1f} "
          f"| sCT HU-MAE ALL {hm(None):.1f} (abd {hm('abdomen'):.1f} / lung {hm('lung'):.1f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
