"""Photon-MRI plan eval scored the CHALLENGE way — the MRI twin of cv_eval_photonct_full.py.

`eval_dose_e2e.py` reconstructs the reference plan by summing CACHED per-CP GT crops, so the metric
inherits the crop margin: a margin-8 cache scores ~9-10 pts optimistic vs the true full-grid GT (the
same bias that fooled us on photon-CT), which makes a margin-8 shipped model and a margin-24 new model
non-comparable. This scores BOTH against the TRUE full-grid GT (scripts/build_gt_plans.py, reused from
photon-CT — the GT dose is MC truth, independent of CT vs MR), each model run through its own DEPLOY
path at its own margin, so new-vs-shipped is clean.

Mirrors the photon-MRI container exactly (container/photon_mri/app.py::_predict_fn): MRI -> (clf coarse
+ E2E synth) sCT density -> photon_mri.predict.predict_cps(E2E.dose) -> plan -> gamma vs true GT.

  python scripts/cv_eval_photonmri_full.py --config CFG --ckpt E2E.pt --clf CLF.pt --out OUT.csv \
      [--margin 24] [--gpu-gamma 10] [--patients a,b,c]
"""
from __future__ import annotations

import argparse, csv, json, os, statistics as st, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True, help="E2E state.pt (synth.* + dose.*)")
ap.add_argument("--clf", default="/data/kwang/sct_classify_runs/clf_whole/best.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=int, default=24)
ap.add_argument("--cutoff", type=float, default=3.328e-7)
ap.add_argument("--gt-plans", default=None)
ap.add_argument("--patients", default=None)
ap.add_argument("--gpu-gamma", type=int, default=10)
a = ap.parse_args()

os.environ["DOSERAD_PHOTON_MARGIN"] = str(a.margin)   # read at import by photon_mri.predict

from doserad.eval.plan_agg import stratified_mae                # noqa: E402
from doserad.eval.gamma import gamma_array, gamma_pass          # noqa: E402
from doserad.physics.machine import load_photon_machine         # noqa: E402
from doserad.io.mha import load_mha                             # noqa: E402
from container.mri_synth import synth_density, load_classifier  # noqa: E402
from container.photon_mri.predict import predict_cps            # noqa: E402
from container.proton.gc_invoke import _apply_cutoff            # noqa: E402
from train_dose_e2e import E2E                                  # noqa: E402

ROOT = Path("/data/kwang/DoseRad2026_raw/photon/training")
WORK = Path(os.environ.get("DOSERAD_WORK", "/home/kaiwang/doserad2026_workdir"))
GT_PLANS = Path(a.gt_plans) if a.gt_plans else WORK / "cache" / "gt_plans" / "photon"

cfg = yaml.safe_load(open(a.config))
dev = "cuda"
val = a.patients.split(",") if a.patients else \
    json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]

machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
net = E2E(cfg).to(dev).eval()
sd = torch.load(a.ckpt, map_location=dev)
net.load_state_dict(sd.get("ema", sd.get("model")))
clf = load_classifier(a.clf, dev)
img_ch = int(cfg.get("in_ch", 6)) > 6
print(f"loaded E2E {a.ckpt} (step {sd.get('step','?')}), clf {a.clf}, margin {a.margin}, img_ch {img_ch}",
      flush=True)


class _Vol:                                    # what predict_cps needs: geometry + density array
    def __init__(self, sitk_img, density_np):
        self.array = density_np
        self.spacing = tuple(sitk_img.GetSpacing())
        self.origin = tuple(sitk_img.GetOrigin())


rows = []
for pid in val:
    gtf = GT_PLANS / f"{pid}.npy"
    if not gtf.exists():
        print(f"  {pid}: no full-grid GT plan ({gtf})", flush=True); continue
    mr_sitk = sitk.ReadImage(str(ROOT / pid / "image" / "mr.mha"))
    plan = json.load(open(ROOT / pid / f"{pid}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi; beams.append(b)

    with torch.no_grad():
        density_np, sct01 = synth_density(mr_sitk, clf, net, dev, density_direct=False,
                                          hu_anchors=machine.hu_anchors)
        vol = _Vol(mr_sitk, density_np)
        density_t = torch.from_numpy(density_np).to(dev)
        preds = predict_cps(vol, beams, density_np, density_t, net.dose, machine, dev, img_ch=img_ch)

    full = density_np.shape
    pp = np.zeros(full, np.float32)
    for crop, bbox in preds.values():
        c = _apply_cutoff(crop, {"minimum_cutoff": a.cutoff}) if a.cutoff > 0 else crop
        z0, z1, y0, y1, x0, x1 = bbox
        pp[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += c

    gt = np.load(gtf).astype(np.float32)
    if gt.shape != full:
        print(f"  {pid}: grid mismatch pred {full} vs GT {gt.shape} — skip", flush=True); continue
    rx = float(gt.max())
    if a.gpu_gamma:
        from doserad.eval.gamma_gpu import gamma_array_gpu
        g1c, g1m = gamma_array_gpu(pp, gt, vol.spacing, rx, 1.0, 1.0, interp_fraction=a.gpu_gamma)
        g3c, g3m = gamma_array_gpu(pp, gt, vol.spacing, rx, 3.0, 3.0, interp_fraction=a.gpu_gamma)
        g3 = float((g3c[g3m] <= 1).mean()) if g3m.any() else float("nan")
    else:
        g1c, g1m = gamma_array(pp, gt, vol.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        g3 = gamma_pass(pp, gt, vol.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    g1 = float((g1c[g1m] <= 1).mean()) if g1m.any() else float("nan")

    row = {"patient": pid, "site": "lung" if "THB" in pid else "abdomen",
           "plan_g1": g1, "plan_g3": g3, "strat_mae": stratified_mae(pp, gt, rx),
           "pred_energy_frac": float(pp.sum() / gt.sum())}
    rows.append(row)
    print(f"  {pid} ({row['site']}): g1 {g1*100:.2f} g3 {g3*100:.2f} energy {row['pred_energy_frac']*100:.1f}%",
          flush=True)

if rows:
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    m = st.mean([r["plan_g1"] for r in rows]) * 100
    ab = [r["plan_g1"] for r in rows if r["site"] == "abdomen"]
    lu = [r["plan_g1"] for r in rows if r["site"] == "lung"]
    print(f"\n[{cfg.get('exp_name','?')} m{a.margin}] TRUE mean g1 {m:.2f} "
          f"(abd {st.mean(ab)*100:.1f} / lung {st.mean(lu)*100:.1f}) over {len(rows)} patients -> {a.out}")
