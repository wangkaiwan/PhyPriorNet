"""sCT gamma on the 6 downloaded proton web-test patients (paired MR+CT), in DOSE space.
For each patient: run the SAME production proton dose net on 3 densities — real-CT (reference),
OLD 2mm sCT (clf_whole_mraug+prod synth), NEW 1x1x3 sCT (clf_1x1x3+ref_1x1x3) — and compute local
gamma(old vs real) and gamma(new vs real). Engine/net error cancels; only the sCT differs.
Answers: does the 1x1x3 sCT give better/worse dose gamma than the 2mm sCT on the real test?"""
import os, sys, glob, json, argparse
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
import numpy as np, SimpleITK as sitk, torch
from monai.inferers import sliding_window_inference
from train_sct_paired import norm_mr, load_arr
import train_sct_classifier as CLF
import train_sct_refiner as REF
from container.proton_mri import app
from container.mri_synth import synth_density
from container.proton.predict import predict_beams
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from doserad.physics.density import hu_to_density

TB = "/data/kwang/doserad_test_inputs/proton_testset"
CT_LO, CT_HI = -1000.0, 2000.0
REP_HU = np.asarray([-1000., -600., 30., 700.], np.float32)

app.load_models()
clf_p = app._STATE["clf"]; net_p = app._STATE["net"]; dd = app._STATE.get("density_direct", True)
pm = app._STATE["pm"]; machine = app._STATE["machine"]; HU = machine.hu_anchors; DEV = app.DEV
dose_net = net_p.dose

# NEW 1x1x3 cascade (env-overridable; default = full-data samefield 182)
_NEW_CLF = os.environ.get("DOSERAD_NEW_CLF", "/data/kwang/sct_classify_runs/clf_1x1x3_samefield/best.pt")
_NEW_REF = os.environ.get("DOSERAD_NEW_REF", "/data/kwang/sct_refine_runs/ref_1x1x3_samefield/best.pt")
clf13 = CLF.model().to(DEV).eval(); clf13.load_state_dict(torch.load(_NEW_CLF, map_location="cpu")["net"])
ref13 = REF.model().to(DEV).eval(); ref13.load_state_dict(torch.load(_NEW_REF, map_location="cpu")["net"])
print(f"[new sCT] clf={_NEW_CLF}\n          ref={_NEW_REF}", flush=True)

_WHOLE_REF = os.environ.get("DOSERAD_WHOLE_REF") == "1"   # ref13 trained whole-image -> infer whole-image
def _pad16(n): return (16 - n % 16) % 16
@torch.no_grad()
def new_sct_density(mr_sitk):
    mr = norm_mr(load_arr_from(mr_sitk)).astype(np.float32)               # (x,y,z)
    x = torch.from_numpy(mr)[None, None].to(DEV)
    with torch.autocast("cuda"):
        if _WHOLE_REF:   # whole-image clf forward — MATCHES the precomputed training coarse (HU-MAE 0.0);
                         # sliding-window here fed the refiner a wrong coarse (HU-MAE 125-192) -> false 0.0 gamma.
            X, Y, Z = x.shape[-3:]
            xp = torch.nn.functional.pad(x, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
            logit = clf13(xp)[..., :X, :Y, :Z]
        else:
            logit = sliding_window_inference(x, (128,128,128), 4, clf13, overlap=0.25, mode="gaussian")
        p = torch.softmax(logit.float(), 1)[0]
        coarse_hu = (p * torch.from_numpy(REP_HU).to(DEV).view(-1,1,1,1)).sum(0).cpu().numpy()
        co01 = np.clip((coarse_hu - CT_LO)/(CT_HI-CT_LO), 0, 1).astype(np.float32)
        xin = torch.from_numpy(np.stack([mr, co01], 0)[None]).to(DEV)
        if _WHOLE_REF:                                   # whole-trained refiner -> whole-image forward
            X, Y, Z = xin.shape[-3:]
            xp = torch.nn.functional.pad(xin, (0, _pad16(Z), 0, _pad16(Y), 0, _pad16(X)))
            pr = ref13(xp)[0, 0, :X, :Y, :Z].float().cpu().numpy()
        else:
            pr = sliding_window_inference(xin, (128,128,128), 2, ref13, overlap=0.25, mode="gaussian")[0,0].float().cpu().numpy()
    sct_hu = (np.clip(pr,0,1)*(CT_HI-CT_LO)+CT_LO).transpose(2,1,0)        # -> (z,y,x)
    return hu_to_density(sct_hu, HU).astype(np.float32)

def load_arr_from(sitk_img):
    return np.transpose(sitk.GetArrayFromImage(sitk_img).astype(np.float32), (2,1,0))

def plan_dose(mr_sitk, density_np):
    img = app._Img(mr_sitk, density_np)
    dt = torch.as_tensor(density_np, device=DEV)
    preds = predict_beams(img, BEAMS, density_np, dt, dose_net, pm, DEV)
    return accumulate_plan([(d, bb) for (d, bb) in preds.values()], density_np.shape)

# ---- pairing (size-grouped, registered) + image->beam-entry mapping ----
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
    mhas = sorted(glob.glob(f"{folder}/*.mha"))
    idx = mhas.index(mf)
    ent = next((e for e in meta if e.get("image_file_idx")==idx), None)
    if ent is None:
        ent = meta[idx] if idx < len(meta) else meta[0]
    return ent["beams"]

mr_by = by_size(f"{TB}/ProtonMRI/set*/*.mha"); ct_by = by_size(f"{TB}/ProtonCT/set*/*.mha")
pairs = []
for sz in sorted(mr_by):
    if sz not in ct_by: continue
    for mf in mr_by[sz]:
        cf = next((c for c in ct_by[sz] if registered(mf, c)), None)
        if cf: pairs.append((mf, cf)); break

ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=99); a = ap.parse_args()
print(f"{len(pairs)} registered test pairs; evaluating {min(a.max,len(pairs))}\n")
rows = []
for i, (mf, cf) in enumerate(pairs[:a.max]):
    tag = f"{os.path.basename(os.path.dirname(mf))}/{os.path.basename(mf)[:8]}"
    mr_sitk = sitk.ReadImage(mf); ct_sitk = sitk.ReadImage(cf)
    BEAMS = beam_entry(mf)
    ct_r = sitk.Resample(ct_sitk, mr_sitk, sitk.Transform(), sitk.sitkLinear, -1000., sitk.sitkFloat32)
    dens_real = hu_to_density(sitk.GetArrayFromImage(ct_r).astype(np.float32), HU).astype(np.float32)
    dens_old, _ = synth_density(mr_sitk, clf_p, net_p, DEV, density_direct=dd, hu_anchors=HU)
    dens_new = new_sct_density(mr_sitk)
    d_real = plan_dose(mr_sitk, dens_real); d_old = plan_dose(mr_sitk, dens_old); d_new = plan_dose(mr_sitk, dens_new)
    sp = mr_sitk.GetSpacing(); rx = float(d_real.max())
    if rx <= 0:
        print(f"[{i}] {tag}: zero reference dose, skip"); continue
    def gpass(test, dpct, dta):
        g, m = gamma_array(test, d_real, sp, rx, dose_pct=dpct, dta_mm=dta)
        return 100.0*float((g[m] <= 1.0).mean()) if m.any() else float("nan")
    r = {"tag": tag,
         "old_2/2": gpass(d_old,2,2), "new_2/2": gpass(d_new,2,2),
         "old_3/3": gpass(d_old,3,3), "new_3/3": gpass(d_new,3,3),
         "self":    gpass(d_real,2,2)}
    rows.append(r)
    print(f"[{i}] {tag:22s} | 2%/2mm old {r['old_2/2']:5.1f} new {r['new_2/2']:5.1f} | 3%/3mm old {r['old_3/3']:5.1f} new {r['new_3/3']:5.1f} | self {r['self']:.0f}", flush=True)

if rows:
    def mean(k): return float(np.nanmean([r[k] for r in rows]))
    print("\n=== MEAN over", len(rows), "patients ===")
    print(f"  2%/2mm:  OLD 2mm sCT {mean('old_2/2'):.1f}   NEW 1x1x3 sCT {mean('new_2/2'):.1f}   Δ {mean('new_2/2')-mean('old_2/2'):+.1f}")
    print(f"  3%/3mm:  OLD 2mm sCT {mean('old_3/3'):.1f}   NEW 1x1x3 sCT {mean('new_3/3'):.1f}   Δ {mean('new_3/3')-mean('old_3/3'):+.1f}")
    print(f"  (self-gamma real-vs-real: {mean('self'):.0f}  <- sanity, should be ~100)")
