"""Internal DVH (official dvh_clinical_score) on held16 — the last unmeasured board metric.

Structures: OARs from TotalSegmentator (cache/totalseg_held16/<pid>/seg/*.nii.gz, CT grid) — the
same provenance the organizers use (preprocessing README). PTV approximated from the GT plan dose
(>= 90% of Rx), identical across candidates -> fair RELATIVE comparison. Official scoring: PTV
D98/V95 + 3 OARs nearest the PTV centroid (D2/Dmean), % of Rx.

Calibration anchors (board DVH): base48@m16 0.3004 | b32old@m16 0.3764 | b32@m12 0.4302.

Usage: DOSERAD_PHOTON_MARGIN=16 DOSERAD_W_OVERRIDE=... CUDA_VISIBLE_DEVICES=0 \
       python scripts/eval_dvh_held16.py <label> [N]
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "cand"
NPID = int(sys.argv[2]) if len(sys.argv) > 2 else 16
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "16")
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = os.environ.get("DOSERAD_W_OVERRIDE", f"{RUNS}/docker_extracted/photon_ct_docker.pt")
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
os.environ["DOSERAD_MODALITY"] = "ct"
import numpy as np
import SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from official_eval.metrics_plan import dvh_clinical_score
from container.photon import app

FROZEN = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))
PIDS = FROZEN["held16"][:NPID]
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
SEG = Path("/home/kaiwang/doserad2026_workdir/cache/totalseg_held16")
# candidate OARs (TotalSeg names); per-patient we keep those with enough voxels, official picks 3 by centroid distance
OAR_NAMES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
             "lung_middle_lobe_right", "lung_lower_lobe_right", "heart", "esophagus",
             "spinal_cord", "liver", "stomach", "kidney_left", "kidney_right",
             "duodenum", "small_bowel", "colon", "spleen", "pancreas", "aorta"]


def rx_of(pid):
    return 70.0 if pid.upper().startswith("1TH") else 60.0


def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        bb = dict(b); bb["beam_idx"] = bi
        for cp in bb["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(bb)
    return {"image_file_idx": 0, "beams": beams}


app.load_models()
print(f"[dvh] label={LABEL} margin={os.environ['DOSERAD_PHOTON_MARGIN']} N={len(PIDS)}", flush=True)
scores = []
for pid in PIDS:
    src = sitk.ReadImage(f"{ROOT}/{pid}/image/ct.mha")
    plan = json.load(open(f"{ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    full = sitk.GetArrayFromImage(src).shape
    pred_cps, gt_cps = [], []
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        z = np.load(f); gt_cps.append((z["dose"].astype(np.float32), tuple(int(v) for v in z["bbox"])))
        if (bi, cpi) in preds:
            pcrop, pbb, _ = preds[(bi, cpi)]; pred_cps.append((pcrop, pbb))
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    rx = rx_of(pid)
    # scale plans to Gy-ish: our plans are raw-summed CP doses; official normalizes by prescription.
    # For RELATIVE candidate comparison scale both so GT max region ~ Rx: s = rx / gt.max() (same for both).
    s = rx / max(float(gt.max()), 1e-9)
    ppd = pp * s; gtd = gt * s
    ptv = (gtd >= 0.90 * rx)
    if ptv.sum() < 100:
        print(f"  {pid}: tiny PTV, skip"); continue
    masks = [ptv]
    names = ["PTV"]
    for nm in OAR_NAMES:
        f = SEG / pid / "seg" / f"{nm}.nii.gz"
        if not f.exists(): continue
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
        if m.shape != gtd.shape or m.sum() < 500: continue
        m = m & ~ptv
        if m.sum() < 500: continue
        masks.append(m); names.append(nm)
    if len(masks) < 4:
        print(f"  {pid}: only {len(masks)-1} OARs, skip"); continue
    structures = np.stack(masks, axis=-1).astype(np.uint8)
    res = dvh_clinical_score(ppd, gtd, structures, names, ptv_index=0,
                             oar_indices=list(range(1, len(names))), prescription_dose=rx)
    scores.append(res["dvh_score"])
    print(f"  {pid}: dvh {res['dvh_score']:.4f}", flush=True)
print(f"\n>>> {LABEL}: DVH mean {np.mean(scores):.4f} over {len(scores)} patients", flush=True)
