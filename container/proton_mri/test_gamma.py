"""Proton-MRI container gamma test: full deploy path (compiled synth+dose, in-container coarse+synth
density_direct RESAMPLED to the native proton grid, geom-bbox + PB-tight-crop) on one patient, plan
gamma vs the MC GT proton dose (training cache). Target: match 5CV baseline (87.4 ALL).

Deployment gives only an MR on the (native proton) reference grid. We mimic that by resampling the
2mm mr.mha to the proton ct.mha grid and feeding THAT as the source image, so the container's
native->2mm->synth->native round-trip is exercised end-to-end.
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/proton_mri/test_gamma.py 1ABB006
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/cv/se_protonmri_f0.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/se_protonmri_f0/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
from pathlib import Path
import numpy as np, SimpleITK as sitk, yaml
from container.proton_mri import app
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

PID = sys.argv[1] if len(sys.argv) > 1 else "1ABB006"
PROTON_ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
cfg = yaml.safe_load(open(os.environ["DOSERAD_CONFIG"]))
cache = Path(cfg["cache_dir"])


def main():
    app.load_models()
    ct_sitk = sitk.ReadImage(f"{PROTON_ROOT}/{PID}/image/ct.mha")        # native proton grid (geometry)
    mr2 = sitk.ReadImage(f"{PHOTON_ROOT}/{PID}/image/mr.mha")            # 2mm MR
    # mimic the deployment source: MR resampled onto the native proton reference grid
    src_mr = sitk.Resample(mr2, ct_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    plan = json.load(open(f"{PROTON_ROOT}/{PID}/{PID}.json"))
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for r in b["rays"]:
            for bl in r["beamlets"]:
                bl["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    entry = {"image_file_idx": 0, "beams": beams}

    t0 = time.time()
    preds = app._predict_fn(src_mr, entry)
    dt = time.time() - t0

    full = sitk.GetArrayFromImage(ct_sitk).shape
    pred_cps, gt_cps = [], []
    for f in sorted(g for g in (cache / PID).glob("B*_R*_L*.npz") if ".tmp" not in g.name):
        b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
        z = np.load(f); bb = tuple(int(v) for v in z["bbox"])
        gt_cps.append((z["dose"].astype(np.float32), bb))
        if (b, r, l) in preds:
            crop, pbb, _ = preds[(b, r, l)]
            pred_cps.append((crop, pbb))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max()); zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    crop = (slice(max(int(zz.min())-m, 0), int(zz.max())+m+1), slice(max(int(yy.min())-m, 0), int(yy.max())+m+1),
            slice(max(int(xx.min())-m, 0), int(xx.max())+m+1))
    sp = ct_sitk.GetSpacing()
    g1c, g1m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    g3c, g3m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=3.0, dta_mm=3.0)
    g3 = float((g3c[g3m] <= 1.0).mean()) * 100 if g3m.any() else float("nan")
    print(f"\n=== PROTON-MRI CONTAINER GAMMA ({PID}) ===")
    print(f"  {len(pred_cps)}/{len(gt_cps)} beamlets predicted, {dt:.1f}s "
          f"({dt/max(len(pred_cps),1)*1000:.0f} ms/beamlet)")
    print(f"  plan gamma 1%/1mm {g1:.1f}%   3%/3mm {g3:.1f}%   (baseline 5CV 87.4)")


if __name__ == "__main__":
    main()
