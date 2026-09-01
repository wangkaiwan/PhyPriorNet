"""Email change #5: the official GT was swapped to a minimum_cutoff-ADHERING version (low-dose tail
removed) instead of the raw simulation output. Our container's _apply_cutoff QUANTISES (round-to-nearest)
sub-cutoff dose rather than ZEROING it — a choice made (gc_invoke.py:426) against the RAW GT, whose
premise ("GT keeps its full continuous tail") change #5 may have INVALIDATED.

This isolates the low-dose HANDLING from model quality by testing a PERFECT prediction (pred == GT) plus
the real model, crossing {pred processed by zero|quant} x {GT hypothesis raw|zero|quant}. Cutoff is applied
PER-CP (mirrors per-beam minimum_cutoff) then summed into the plan, then official plan gamma + IDD.

Decision: if the official cut-GT ZEROS below cutoff, the "vs GT-zero" column tells us whether our output
should ZERO (matches GT) instead of QUANTISE, and the perfect-prediction row bounds the gamma at stake.

Usage: CUDA_VISIBLE_DEVICES="" python scripts/diag_cutoff_match.py [N]   (CPU, won't disturb training GPUs)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("DOSERAD_PHOTON_MARGIN", "24")
RUNS = "/home/kaiwang/doserad2026_workdir/runs"
os.environ["DOSERAD_WEIGHTS"] = f"{RUNS}/docker_extracted/photon_ct_docker.pt"
os.environ["DOSERAD_MACHINE"] = "/data/kwang/DoseRad2026_raw/photon/training/beam_parameters.json"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CUTOFFS = [0.005, 0.01, 0.02]                        # per-CP cutoff as fraction of that CP's GT max
PIDS = json.load(open("/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"))["held16"]
PIDS = PIDS[:2] + PIDS[8:8 + max(0, N - 2)]          # 2 abd + (N-2) lung

import numpy as np, SimpleITK as sitk
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from container.photon import app

def build_entry(plan):
    beams = []
    for bi, b in enumerate(plan["beams"]):
        b["beam_idx"] = bi
        for cp in b["control_points"]:
            cp["output_info"] = {"output_file_idx": 0, "idx_in_output": 0, "minimum_cutoff": 0.0}
        beams.append(b)
    return {"image_file_idx": 0, "beams": beams}

def zero(x, cut):   return np.where(x < cut, np.float32(0.0), x).astype(np.float32)
def quant(x, cut):  # container _apply_cutoff: <cut/2 -> 0, [cut/2,cut) -> cut, else x
    return np.where(x < cut/2, np.float32(0.0), np.where(x < cut, np.float32(cut), x)).astype(np.float32)

def gamma_of(pred_cps, gt_cps, full, sp, cr):
    pp = accumulate_plan(pred_cps, full); gt = accumulate_plan(gt_cps, full)
    gmax = float(gt.max())
    if gmax <= 0: return float("nan")
    gc, gm = gamma_array(pp[cr], gt[cr], sp, gmax, dose_pct=1.0, dta_mm=1.0)
    return float((gc[gm] <= 1.0).mean()) * 100 if gm.any() else float("nan")

app.load_models()
print(f"[cutoff-match] N={len(PIDS)}  cutoffs(%CPmax)={[c*100 for c in CUTOFFS]}  (pred==GT rows isolate HANDLING)", flush=True)
# accumulator: keyed by (row-label) -> list over patients, one dict per cutoff
ROWS = ["PPzero_vs_GTraw", "PPquant_vs_GTraw",              # reproduce docstring (90.81 / 99.96)
        "PPquant_vs_GTzero", "PPzero_vs_GTquant",           # perfect-pred MISMATCH cost
        "real_raw_vs_GTraw",                                # current held16 baseline (cutoff-independent)
        "real_zero_vs_GTzero", "real_quant_vs_GTzero",      # if official GT zeros: which output wins?
        "real_zero_vs_GTquant", "real_quant_vs_GTquant"]    # if official GT quantises: which output wins?
agg = {r: {c: [] for c in CUTOFFS} for r in ROWS}
base_list = []

for pid in PIDS:
    src = sitk.ReadImage(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
    full = sitk.GetArrayFromImage(src).shape; sp = src.GetSpacing()
    plan = json.load(open(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    preds = app._predict_fn(src, build_entry(plan))
    P_raw, G_raw = [], []                                    # (crop, bbox) lists, per CP
    for f in sorted((CACHE / pid).glob("*.npz")):
        if ".tmp" in f.name: continue
        bi, cpi = (int(x) for x in f.stem.split("_"))
        if (bi, cpi) not in preds: continue
        z = np.load(f); gbb = tuple(int(v) for v in z["bbox"]); gt = z["dose"].astype(np.float32)
        pcrop, pbb, _ = preds[(bi, cpi)]
        G_raw.append((gt, gbb)); P_raw.append((pcrop.astype(np.float32), pbb))
    # crop region for gamma from raw-GT plan (fixed across all variants)
    gtplan = accumulate_plan(G_raw, full); gmax = float(gtplan.max())
    zz, yy, xx = np.where(gtplan >= 0.05 * gmax); m = 4
    cr = (slice(max(int(zz.min())-m,0), int(zz.max())+m+1), slice(max(int(yy.min())-m,0), int(yy.max())+m+1),
          slice(max(int(xx.min())-m,0), int(xx.max())+m+1))
    # cutoff-independent baseline
    base = gamma_of(P_raw, G_raw, full, sp, cr); base_list.append(base)  # cutoff-independent baseline
    for c in CUTOFFS:
        cuts = [c * float(g.max()) for g, _ in G_raw]        # per-CP cutoff on GT scale
        Gz = [(zero(g, k), bb) for (g, bb), k in zip(G_raw, cuts)]
        Gq = [(quant(g, k), bb) for (g, bb), k in zip(G_raw, cuts)]
        Pz = [(zero(p, k), bb) for (p, bb), k in zip(P_raw, cuts)]
        Pq = [(quant(p, k), bb) for (p, bb), k in zip(P_raw, cuts)]
        PPz = [(zero(g, k), bb) for (g, bb), k in zip(G_raw, cuts)]   # perfect pred == GT, then zeroed
        PPq = [(quant(g, k), bb) for (g, bb), k in zip(G_raw, cuts)]  # perfect pred == GT, then quantised
        agg["PPzero_vs_GTraw"][c].append(  gamma_of(PPz, G_raw, full, sp, cr))
        agg["PPquant_vs_GTraw"][c].append( gamma_of(PPq, G_raw, full, sp, cr))
        agg["PPquant_vs_GTzero"][c].append(gamma_of(PPq, Gz,    full, sp, cr))
        agg["PPzero_vs_GTquant"][c].append(gamma_of(PPz, Gq,    full, sp, cr))
        agg["real_zero_vs_GTzero"][c].append(  gamma_of(Pz, Gz, full, sp, cr))
        agg["real_quant_vs_GTzero"][c].append( gamma_of(Pq, Gz, full, sp, cr))
        agg["real_zero_vs_GTquant"][c].append( gamma_of(Pz, Gq, full, sp, cr))
        agg["real_quant_vs_GTquant"][c].append(gamma_of(Pq, Gq, full, sp, cr))
    print(f"  {pid}: baseline(raw/raw) gamma {base:.2f}", flush=True)

print(f"\n=== PERFECT-PREDICTION (pred==GT, isolates cutoff HANDLING) ===", flush=True)
print(f"  baseline pred-raw vs GT-raw (current held16, cutoff-independent): {np.nanmean(base_list):.2f}", flush=True)
for r in ["PPzero_vs_GTraw", "PPquant_vs_GTraw", "PPquant_vs_GTzero", "PPzero_vs_GTquant"]:
    print(f"  {r:22s}: " + "  ".join(f"C{c*100:.1f}%={np.nanmean(agg[r][c]):6.2f}" for c in CUTOFFS), flush=True)
print(f"\n=== REAL MODEL (4018f597 @ m24) ===", flush=True)
for r in ["real_zero_vs_GTzero", "real_quant_vs_GTzero", "real_zero_vs_GTquant", "real_quant_vs_GTquant"]:
    print(f"  {r:22s}: " + "  ".join(f"C{c*100:.1f}%={np.nanmean(agg[r][c]):6.2f}" for c in CUTOFFS), flush=True)
print(f"\n>>> DECISION: if official GT ZEROS below cutoff, compare real_zero_vs_GTzero (switch to zero) vs "
      f"real_quant_vs_GTzero (our current output). Bigger gap = more gamma we're leaving on the table.", flush=True)
