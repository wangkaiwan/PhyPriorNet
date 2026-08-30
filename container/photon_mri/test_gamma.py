"""Photon-MRI container gamma test: run the FULL deploy path (compiled synth+dose, in-container
coarse+synth density, geometry-bbox channels) on one patient and score plan gamma vs the MC GT
(from the training cache). Target: match the 5CV baseline (91.1 ALL; 1ABB006 abd ~91.4).
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/photon_mri/test_gamma.py 1ABB006
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/cv/se_photonmri_f0.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/se_photonmri_f0/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json")
from pathlib import Path
import numpy as np, SimpleITK as sitk, yaml
from container.photon_mri import app
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
cfg = yaml.safe_load(open(os.environ["DOSERAD_CONFIG"]))
cache = Path(cfg["cache_dir"])


def main():
    app.load_models()
    mr_sitk = sitk.ReadImage(f"{ROOT}/{PID}/image/mr.mha")
    plan = json.load(open(f"{ROOT}/{PID}/{PID}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for ci, cp in enumerate(b["control_points"]):
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    entry = {"image_file_idx": 0, "beams": beams}

    t0 = time.time()
    preds = app._predict_fn(mr_sitk, entry)
    dt = time.time() - t0

    # GT plan from cache (per-CP dose + bbox). Key preds by (beam_idx, cp_idx); cache file stem = "<bi>_<cpidx>"
    full = sitk.GetArrayFromImage(mr_sitk).shape
    pred_cps, gt_cps = [], []
    for f in sorted((cache / PID).glob("*.npz")):
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); bb = tuple(int(v) for v in z["bbox"])
        gt_cps.append((z["dose"].astype(np.float32), bb))
        if (bi, cpi) in preds:
            crop, pbb, _ = preds[(bi, cpi)]
            pred_cps.append((crop, pbb))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max()); zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    crop = (slice(max(int(zz.min())-m, 0), int(zz.max())+m+1), slice(max(int(yy.min())-m, 0), int(yy.max())+m+1),
            slice(max(int(xx.min())-m, 0), int(xx.max())+m+1))
    sp = mr_sitk.GetSpacing()
    g1c, g1m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    g3c, g3m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=3.0, dta_mm=3.0)
    g3 = float((g3c[g3m] <= 1.0).mean()) * 100 if g3m.any() else float("nan")
    print(f"\n=== PHOTON-MRI CONTAINER GAMMA ({PID}) ===")
    print(f"  {len(pred_cps)}/{len(gt_cps)} CPs predicted, {dt:.1f}s ({dt/max(len(pred_cps),1)*1000:.0f} ms/CP)")
    print(f"  plan gamma 1%/1mm {g1:.1f}%   3%/3mm {g3:.1f}%   (baseline 5CV 91.1)")


if __name__ == "__main__":
    main()
