"""Controlled sCT comparison in DOSE space on the 6 proton web-test patients (paired MR+CT):
  Model A (both steps SAME-field): clf_whole_samefield_aug + ref_samefield_ctrl
  Model B (both steps ALL-field):  clf_whole              + ref_allfield_wt2
Both sCTs are built via container.mri_synth.synth_density (the EXACT 2mm deploy path: source MR -> 2mm ->
WHOLE-image clf coarse -> refiner -> sCT density -> resample to source). 2mm clf is whole-image = matches
the whole-image training coarse of both refiners (no sliding-window mismatch). The SAME production proton
dose net runs on real-CT (reference), A, and B -> engine error cancels, only the sCT differs.
Sanity: self (real-vs-real) must be ~100; A/B are the field-recipe comparison.
Usage: CUDA_VISIBLE_DEVICES=1 conda run -n doserad python scripts/eval_sct_AB.py [--max N]
Env override: DOSERAD_A_CLF/DOSERAD_A_REF/DOSERAD_B_CLF/DOSERAD_B_REF.
"""
import os, sys, glob, json, argparse
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "scripts")
import numpy as np, SimpleITK as sitk, torch
import torch.nn.functional as F
import train_sct_refiner as REF
def _pad16(n): return (16 - n % 16) % 16
from container.mri_synth import synth_density, load_classifier
from container.proton_mri import app
from container.proton.predict import predict_beams
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from doserad.physics.density import hu_to_density

TB = "/data/kwang/doserad_test_inputs/proton_testset"
app.load_models()
pm = app._STATE["pm"]; HU = app._STATE["machine"].hu_anchors; DEV = app.DEV
dose_net = app._STATE["net"].dose; dd = app._STATE.get("density_direct", True)

A_CLF = os.environ.get("DOSERAD_A_CLF", "/data/kwang/sct_classify_runs/clf_whole_samefield_aug/best.pt")
A_REF = os.environ.get("DOSERAD_A_REF", "/data/kwang/sct_refine_runs/ref_samefield_ctrl/best.pt")
B_CLF = os.environ.get("DOSERAD_B_CLF", "/data/kwang/sct_classify_runs/clf_whole/best.pt")
B_REF = os.environ.get("DOSERAD_B_REF", "/data/kwang/sct_refine_runs/ref_allfield_wt2/best.pt")

class RefWrap:   # synth_density calls synth.sct01([mr01,co][None]) -> (1,1,z,y,x); our refiner is a raw UNet.
    # Replicate E2E.sct01: pad X,Y,Z to a multiple of 16 (UNet 4-level needs it) then crop back.
    def __init__(self, ref): self.ref = ref
    def sct01(self, x):
        Z, Y, X = x.shape[-3:]
        xp = F.pad(x, (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
        return self.ref(xp)[..., :Z, :Y, :X]

def load_ref(p):
    net = REF.model().to(DEV).eval(); net.load_state_dict(torch.load(p, map_location="cpu")["net"]); return RefWrap(net)

clf_A = load_classifier(A_CLF, DEV); ref_A = load_ref(A_REF)
clf_B = load_classifier(B_CLF, DEV); ref_B = load_ref(B_REF)
print(f"[A same-field] clf={A_CLF}\n               ref={A_REF}")
print(f"[B all-field]  clf={B_CLF}\n               ref={B_REF}", flush=True)

def by_size(pat):
    d = {}
    for f in glob.glob(pat):
        if os.path.getsize(f) < 5e6: continue
        d.setdefault(sitk.ReadImage(f).GetSize(), []).append(f)
    return d
def registered(mf, cf):
    mi = sitk.ReadImage(mf); ci = sitk.ReadImage(cf)
    return (mi.GetSize()==ci.GetSize() and np.allclose(mi.GetOrigin(), ci.GetOrigin(), atol=2)
            and np.allclose(mi.GetDirection(), ci.GetDirection(), atol=1e-2))
def beam_entry(mf):
    folder = os.path.dirname(mf)
    meta = json.load(open(glob.glob(f"{folder}/stacked-proton-beam-level-metadata.*")[0]))
    mhas = sorted(glob.glob(f"{folder}/*.mha")); idx = mhas.index(mf)
    ent = next((e for e in meta if e.get("image_file_idx")==idx), None) or (meta[idx] if idx < len(meta) else meta[0])
    return ent["beams"]

def plan_dose(mr_sitk, density_np, BEAMS):
    img = app._Img(mr_sitk, density_np); dt = torch.as_tensor(density_np, device=DEV)
    preds = predict_beams(img, BEAMS, density_np, dt, dose_net, pm, DEV)
    return accumulate_plan([(d, bb) for (d, bb) in preds.values()], density_np.shape)

mr_by = by_size(f"{TB}/ProtonMRI/set*/*.mha"); ct_by = by_size(f"{TB}/ProtonCT/set*/*.mha")
pairs = []
for sz in sorted(mr_by):
    if sz not in ct_by: continue
    for mf in mr_by[sz]:
        cf = next((c for c in ct_by[sz] if registered(mf, c)), None)
        if cf: pairs.append((mf, cf)); break
ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=99); a = ap.parse_args()
print(f"{len(pairs)} registered test pairs; evaluating {min(a.max,len(pairs))}\n", flush=True)
rows = []
for i, (mf, cf) in enumerate(pairs[:a.max]):
  tag = f"{os.path.basename(os.path.dirname(mf))}/{os.path.basename(mf)[:8]}"
  try:
    mr_sitk = sitk.ReadImage(mf); ct_sitk = sitk.ReadImage(cf); BEAMS = beam_entry(mf)
    ct_r = sitk.Resample(ct_sitk, mr_sitk, sitk.Transform(), sitk.sitkLinear, -1000., sitk.sitkFloat32)
    dens_real = hu_to_density(sitk.GetArrayFromImage(ct_r).astype(np.float32), HU).astype(np.float32)
    dens_A, _ = synth_density(mr_sitk, clf_A, ref_A, DEV, density_direct=dd, hu_anchors=HU)
    dens_B, _ = synth_density(mr_sitk, clf_B, ref_B, DEV, density_direct=dd, hu_anchors=HU)
    d_real = plan_dose(mr_sitk, dens_real, BEAMS); d_A = plan_dose(mr_sitk, dens_A, BEAMS); d_B = plan_dose(mr_sitk, dens_B, BEAMS)
    sp = mr_sitk.GetSpacing(); rx = float(d_real.max())
    if rx <= 0: print(f"[{i}] {tag}: zero reference dose, skip"); continue
    def gpass(test, dpct, dta):
        g, m = gamma_array(test, d_real, sp, rx, dose_pct=dpct, dta_mm=dta)
        return 100.0*float((g[m] <= 1.0).mean()) if m.any() else float("nan")
    r = {"tag": tag, "A_1/1": gpass(d_A,1,1), "B_1/1": gpass(d_B,1,1),
         "A_2/2": gpass(d_A,2,2), "B_2/2": gpass(d_B,2,2), "self": gpass(d_real,2,2)}
    rows.append(r)
    print(f"[{i}] {tag:22s} | 1%/1mm A {r['A_1/1']:5.1f} B {r['B_1/1']:5.1f} | 2%/2mm A {r['A_2/2']:5.1f} B {r['B_2/2']:5.1f} | self {r['self']:.0f}", flush=True)
  except Exception as e:
    print(f"[{i}] {tag}: SKIP ({type(e).__name__}: {e})", flush=True)
if rows:
    def mean(k): return float(np.nanmean([r[k] for r in rows]))
    print(f"\n=== MEAN over {len(rows)} patients (self={mean('self'):.0f}, must be ~100) ===")
    print(f"  1%/1mm:  A same-field {mean('A_1/1'):.1f}   B all-field {mean('B_1/1'):.1f}   Δ(B-A) {mean('B_1/1')-mean('A_1/1'):+.1f}")
    print(f"  2%/2mm:  A same-field {mean('A_2/2'):.1f}   B all-field {mean('B_2/2'):.1f}   Δ(B-A) {mean('B_2/2')-mean('A_2/2'):+.1f}")
