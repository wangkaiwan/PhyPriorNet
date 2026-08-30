"""Baseline the 4 DEPLOYED docker models on the FROZEN held16 cohort, via the EXACT container deploy
path (loads staged-equivalent weights through each container app), scoring plan gamma vs MC GT cache.

One task per process (env + compiled model are per-task). Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n doserad python scripts/eval_deploy_held16.py <task>
    task = photon_ct | photon_mri | proton_ct | proton_mri

Cohort is read from eval_cohort_frozen.json (held16 = 8 abd 1ABB* + 8 lung 1THB*). NOTE: all 4 models
are FULL-DATA (all75/m24) so held16 is IN-SAMPLE — this is a consistent internal reference baseline,
NOT held-out. Writes runs/baseline_held16_<task>.csv + prints ALL/abd/lung γ1/1 and γ3/3.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

TASK = sys.argv[1]
FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))
PIDS = FROZEN["held16"]
if os.environ.get("N"):  # smoke: first N patients only
    PIDS = PIDS[:int(os.environ["N"])]
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
CLF = "/data/kwang/sct_classify_runs/clf_whole/best.pt"
MACH_PROTON = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
MACH_PHOTON = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PROTON_ROOT = "/data/kwang/DoseRad2026_raw/proton/training"

CFG = {
    "photon_ct": dict(
        # ACTUAL doserad-photon:p2 image weight (md5 4018f597) — NOT all75_p2_ftg. GT = live m24 cache
        # (margin-24, matches DOSERAD_PHOTON_MARGIN=24 the image ships). photon-CT & photon-MRI share
        # the same MC photon-dose GT.
        weights=f"{RUNS}/docker_extracted/photon_ct_docker.pt", label="docker:p2(4018f597)",
        cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24",
        machine=MACH_PHOTON, proton=False),
    "photon_mri": dict(
        # ACTUAL doserad-photon-mri:scheme2p4 image weight (md5 7e05dbdc = m24S2_p4_mmB) + clf_whole.
        weights=f"{RUNS}/docker_extracted/photon_mri_docker.pt", label="docker:scheme2-p4",
        cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24",
        config="configs/experiments/all75/m24S2_p4_mmB.yaml",
        machine=MACH_PHOTON, proton=False),
    "proton_ct": dict(
        weights=f"{RUNS}/all75_r2_ft/state.pt", label="all75_r2_ft",
        cache="/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd",
        machine=MACH_PROTON, proton=True),
    "proton_mri": dict(
        weights=f"{RUNS}/all75_r3_protonmri/state.pt", label="all75_r3",
        cache="/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd",
        config="configs/experiments/cv/se_protonmri_f0.yaml",
        machine=MACH_PROTON, proton=True),
}[TASK]

# --- set env BEFORE importing the container app (apps read env at import/load) ---
# photon apps use torch.compile; inductor codegen fails on this local sm_120 GPU, so force eager
# (the photon app's own docstring: compiled==eager to 1.2e-3 → gamma identical). proton runs eager already.
if not CFG["proton"]:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # CRITICAL: the photon dockers ship DOSERAD_PHOTON_MARGIN=24 (Dockerfile) to capture the ~4% of
    # scatter-tail energy that falls outside margin-8. The GT cache is margin-24 (photon_skinentry_m24).
    # Default is 8 → pred under-crops vs GT → depressed low-band 1% gamma. Must match the docker.
    os.environ["DOSERAD_PHOTON_MARGIN"] = "24"
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", CFG["weights"])
os.environ["DOSERAD_MACHINE"] = CFG["machine"]
if "config" in CFG:
    os.environ["DOSERAD_CONFIG"] = os.environ.get("DOSERAD_CONFIG_OVERRIDE", CFG["config"])
    os.environ["DOSERAD_CLF"] = os.environ.get("DOSERAD_CLF_OVERRIDE", CLF)
IS_MRI = "config" in CFG and TASK.endswith("mri")

import numpy as np
import SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array

if TASK == "photon_ct":
    from container.photon import app
elif TASK == "photon_mri":
    from container.photon_mri import app
elif TASK == "proton_ct":
    os.environ["DOSERAD_MODALITY"] = "ct"
    from container.proton import app
elif TASK == "proton_mri":
    from container.proton_mri import app
else:
    raise SystemExit(f"unknown task {TASK}")

CACHE = Path(CFG["cache"])
_MR_SHIFT = os.environ.get("DOSERAD_MR_SHIFT")   # apply cross-institution MR intensity shift to the source MR


def _apply_mr_shift(a_mr):
    """Same shift as eval_protonmri_plan._apply_mr_shift: smooth bias field + post-norm gamma that both
    survive the container's internal pct-1/99 norm. amp=DOSERAD_SHIFT_BIAS, gam=DOSERAD_SHIFT_GAMMA."""
    from scipy.ndimage import gaussian_filter, zoom
    amp = float(os.environ.get("DOSERAD_SHIFT_BIAS", "0.25"))
    gam = float(os.environ.get("DOSERAD_SHIFT_GAMMA", "1.25"))
    rng = np.random.default_rng(0)
    small = rng.normal(0, 1, [max(s // 8, 2) for s in a_mr.shape]).astype(np.float32)
    small = gaussian_filter(small, 2.0)
    f = zoom(small, [a_mr.shape[i] / small.shape[i] for i in range(3)], order=1)
    bias = 1.0 + amp * (f / (np.abs(f).max() + 1e-6))
    ap = a_mr * bias
    lo, hi = np.percentile(ap, 1), np.percentile(ap, 99)
    n = np.clip((ap - lo) / max(hi - lo, 1.0), 0, 1)
    return (np.clip(n, 0, None) ** gam) * (hi if hi > 0 else 1.0)


def _shift_sitk(mr_sitk):
    if not _MR_SHIFT:
        return mr_sitk
    a = _apply_mr_shift(sitk.GetArrayFromImage(mr_sitk).astype(np.float32))
    out = sitk.GetImageFromArray(a); out.CopyInformation(mr_sitk)
    return out


def build_entry(plan, proton):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        if proton:
            for ri, r in enumerate(b["rays"]):
                r["ray_idx"] = ri
                for bl in r["beamlets"]:
                    bl["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        else:
            for cp in b["control_points"]:
                cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}


def load_input(pid):
    """returns (src_image_for_predict, ref_sitk_for_grid, plan_root)."""
    if TASK == "photon_ct":
        ct = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
        return ct, ct, PHOTON_ROOT
    if TASK == "photon_mri":
        mr = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/mr.mha")
        return _shift_sitk(mr), mr, PHOTON_ROOT
    if TASK == "proton_ct":
        ct = sitk.ReadImage(f"{PROTON_ROOT}/{pid}/image/ct.mha")
        return ct, ct, PROTON_ROOT
    if TASK == "proton_mri":
        ct = sitk.ReadImage(f"{PROTON_ROOT}/{pid}/image/ct.mha")        # native proton grid (geometry)
        mr2 = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/mr.mha")       # 2mm MR
        src = sitk.Resample(mr2, ct, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        return _shift_sitk(src), ct, PROTON_ROOT


def gather_cps(pid, preds, proton):
    pred_cps, gt_cps = [], []
    if proton:
        files = sorted(g for g in (CACHE / pid).glob("B*_R*_L*.npz") if ".tmp" not in g.name)
        for f in files:
            b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
            z = np.load(f); bb = tuple(int(v) for v in z["bbox"])
            gt_cps.append((z["dose"].astype(np.float32), bb))
            if (b, r, l) in preds:
                crop, pbb, _ = preds[(b, r, l)]
                pred_cps.append((crop, pbb))
    else:
        for f in sorted((CACHE / pid).glob("*.npz")):
            if ".tmp" in f.name:
                continue
            bi, cpi = (int(x) for x in f.stem.split("_"))
            z = np.load(f); bb = tuple(int(v) for v in z["bbox"])
            gt_cps.append((z["dose"].astype(np.float32), bb))
            if (bi, cpi) in preds:
                crop, pbb, _ = preds[(bi, cpi)]
                pred_cps.append((crop, pbb))
    return pred_cps, gt_cps


def score(pid):
    src, ref, root = load_input(pid)
    plan = json.load(open(f"{root}/{pid}/{pid}.json"))
    entry = build_entry(plan, CFG["proton"])
    t0 = time.time()
    preds = app._predict_fn(src, entry)
    dt = time.time() - t0
    full = sitk.GetArrayFromImage(ref).shape
    pred_cps, gt_cps = gather_cps(pid, preds, CFG["proton"])
    pp = accumulate_plan(pred_cps, full)
    gt = accumulate_plan(gt_cps, full)
    rx = float(gt.max())
    zz, yy, xx = np.where(gt >= 0.05 * rx); m = 4
    crop = (slice(max(int(zz.min()) - m, 0), int(zz.max()) + m + 1),
            slice(max(int(yy.min()) - m, 0), int(yy.max()) + m + 1),
            slice(max(int(xx.min()) - m, 0), int(xx.max()) + m + 1))
    sp = ref.GetSpacing()
    g1c, g1m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1.0).mean()) * 100 if g1m.any() else float("nan")
    g3c, g3m = gamma_array(pp[crop], gt[crop], sp, rx, dose_pct=3.0, dta_mm=3.0)
    g3 = float((g3c[g3m] <= 1.0).mean()) * 100 if g3m.any() else float("nan")
    return g1, g3, len(pred_cps), len(gt_cps), dt


def main():
    print(f"[{TASK}] weights={CFG['weights']} ({CFG['label']}) cache={CFG['cache']}")
    app.load_models()
    out = Path(f"{RUNS}/baseline_held16_{TASK}.csv")
    rows = [("pid", "site", "g1_1", "g3_3", "npred", "ngt", "sec")]
    for pid in PIDS:
        site = "lung" if "THB" in pid else "abdomen"
        try:
            g1, g3, npd, ngt, dt = score(pid)
            print(f"  {pid} ({site}): g1/1 {g1:5.1f}  g3/3 {g3:5.1f}  [{npd}/{ngt} cp, {dt:.0f}s]", flush=True)
            rows.append((pid, site, f"{g1:.2f}", f"{g3:.2f}", npd, ngt, f"{dt:.1f}"))
        except Exception as e:  # noqa: BLE001
            print(f"  {pid} ({site}): ERROR {type(e).__name__}: {e}", flush=True)
            rows.append((pid, site, "nan", "nan", 0, 0, "0"))
    import csv
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    def agg(sub):
        vals = [float(r[2]) for r in rows[1:] if r[2] != "nan" and (sub is None or r[1] == sub)]
        return sum(vals) / len(vals) if vals else float("nan")

    def agg3(sub):
        vals = [float(r[3]) for r in rows[1:] if r[3] != "nan" and (sub is None or r[1] == sub)]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n=== {TASK} ({CFG['label']}) held16 baseline [IN-SAMPLE, full-data model] ===")
    print(f"  γ1/1 ALL {agg(None):.2f}  (abd {agg('abdomen'):.2f} / lung {agg('lung'):.2f})")
    print(f"  γ3/3 ALL {agg3(None):.2f}  (abd {agg3('abdomen'):.2f} / lung {agg3('lung'):.2f})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
