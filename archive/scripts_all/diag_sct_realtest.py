"""Experiment 1: sCT quality on the REAL test set (paired MR+CT) vs training baseline.
Measures how much our sCT front-end degrades on the unseen test patients — the proton-MRI gap proxy.
No dose GT needed. See docs/superpowers/specs/2026-08-02-protonmri-gap-diagnostic-design.md"""
import os, sys, glob
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, SimpleITK as sitk, torch
from container.proton_mri import app
from container.mri_synth import synth_density
from doserad.physics.density import hu_to_density

app.load_models()
clf = app._STATE["clf"]; synth = app._STATE["net"]; dd = app._STATE["density_direct"]
HU_ANCHORS = app._STATE["machine"].hu_anchors
DEV = app.DEV
BANDS = [("air", -1e9, -500), ("lung", -500, -150), ("soft", -150, 150), ("bone", 150, 1e9)]

def big(folder):
    fs = [f for f in glob.glob(f"{folder}/*.mha") if os.path.getsize(f) > 5e6]
    return max(fs, key=os.path.getsize) if fs else None

def eval_pair(mr_path, ct_path):
    mr = sitk.ReadImage(mr_path); ct = sitk.ReadImage(ct_path)
    # sCT density from MR (container path), on the MR/source grid
    dens_sct, _ = synth_density(mr, clf, synth, DEV, dd)          # np (z,y,x) on source grid
    # real CT -> density, resampled to the sCT (source) grid
    ct_r = sitk.Resample(ct, mr, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    hu = sitk.GetArrayFromImage(ct_r).astype(np.float32)
    dens_ct = hu_to_density(hu, HU_ANCHORS).astype(np.float32)
    body = hu > -500                                              # in-body
    mrarr = sitk.GetArrayFromImage(mr).astype(np.float32)
    mrbody = mrarr > np.percentile(mrarr, 60)
    overlap = (body & mrbody).sum() / max(mrbody.sum(), 1)
    if overlap < 0.75:                                            # misaligned pair -> skip (registration artifact)
        print(f"    SKIP {os.path.basename(os.path.dirname(mr_path))} overlap {overlap*100:.0f}%", flush=True)
        return None
    ad = np.abs(dens_sct - dens_ct)
    row = {"dens_MAE": float(ad[body].mean())}
    for nm, lo, hi in BANDS:
        m = (hu >= lo) & (hu < hi)
        row[nm] = float(ad[m].mean()) if m.sum() > 100 else float("nan")
    # WEPL proxy: integrate density along z (axial range proxy), MAE over the in-body footprint
    wsct = dens_sct.sum(0); wct = dens_ct.sum(0)
    fp = (hu > -500).any(0)
    row["WEPL_MAE_g/cm2proxy"] = float(np.abs(wsct - wct)[fp].mean())
    return row

def summarize(name, pairs):
    rows = [r for m,c in pairs if (r:=eval_pair(m,c)) is not None]
    keys = rows[0].keys()
    print(f"\n=== {name} (n={len(rows)}) ===")
    for k in keys:
        v = [r[k] for r in rows if not np.isnan(r[k])]
        print(f"  {k:22s} {np.mean(v):.4f}")
    return {k: np.nanmean([r[k] for r in rows]) for k in keys}

# TEST: group ALL ProtonMRI/ProtonCT files by size (each size = 1 patient); pair registered MR+CT
TB = "/data/kwang/doserad_test_inputs/proton_testset"
def by_size(pat):
    d = {}
    for f in glob.glob(pat):
        if os.path.getsize(f) < 5e6: continue
        d.setdefault(sitk.ReadImage(f).GetSize(), []).append(f)
    return d
mr_by = by_size(f"{TB}/ProtonMRI/set*/*.mha"); ct_by = by_size(f"{TB}/ProtonCT/set*/*.mha")
def registered(mf, cf):
    mi = sitk.ReadImage(mf); ci = sitk.ReadImage(cf)
    return (mi.GetSize()==ci.GetSize()
            and np.allclose(mi.GetOrigin(), ci.GetOrigin(), atol=2)
            and np.allclose(mi.GetDirection(), ci.GetDirection(), atol=1e-2))
test_pairs = []
for sz in sorted(mr_by):
    if sz not in ct_by: continue
    for mf in mr_by[sz]:
        cf = next((c for c in ct_by[sz] if registered(mf, c)), None)
        if cf: test_pairs.append((mf, cf)); break
print(f"distinct test patients (size-grouped, registered): {len(test_pairs)}")

# TRAIN baseline: proton training paired ct.mha + mr.mha
tr = sorted(glob.glob("/data/kwang/DoseRad2026_raw/proton/training/1*/image"))[:8]
train_pairs = [(f"{d}/mr.mha", f"{d}/ct.mha") for d in tr if os.path.exists(f"{d}/mr.mha") and os.path.exists(f"{d}/ct.mha")]

t = summarize("REAL TEST (unseen)", test_pairs)
r = summarize("TRAINING (seen)", train_pairs)
print("\n=== GENERALIZATION GAP (test - train) ===")
for k in t: print(f"  {k:22s} test {t[k]:.4f}  train {r[k]:.4f}  Δ {t[k]-r[k]:+.4f}")
