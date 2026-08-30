"""Diagnose the ERROR-SPACE STRUCTURE of the PRODUCTION (2mm) sCT on the 6 web-test pairs (real CT = truth).
Goal: decide which loss to add next. Break the sCT density error down by:
  (1) tissue class (air/lung/soft/bone): signed + abs  -> is it a per-class bias (dead) or spread?
  (2) boundary vs interior (dilated class edges)         -> boundary-localized => GAN/perceptual/gradient loss
  (3) per-patient (variance across the 6)                -> a few patients much worse => OOD => data/domain-adapt
No training. Uses the same production sCT (clf_whole_mraug + all75_r3ft_mraug) as deployment."""
import os, sys, glob, json
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, SimpleITK as sitk, torch
from scipy import ndimage
from container.proton_mri import app
from container.mri_synth import synth_density
from doserad.physics.density import hu_to_density

TB = "/data/kwang/doserad_test_inputs/proton_testset"
app.load_models()
clf_p = app._STATE["clf"]; net_p = app._STATE["net"]; dd = app._STATE.get("density_direct", True)
HU = app._STATE["machine"].hu_anchors; DEV = app.DEV
CLASSES = [("air", -1e9, -500), ("lung", -500, -150), ("soft", -150, 150), ("bone", 150, 1e9)]

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
mr_by = by_size(f"{TB}/ProtonMRI/set*/*.mha"); ct_by = by_size(f"{TB}/ProtonCT/set*/*.mha")
pairs = []
for sz in sorted(mr_by):
    if sz not in ct_by: continue
    for mf in mr_by[sz]:
        cf = next((c for c in ct_by[sz] if registered(mf, c)), None)
        if cf: pairs.append((mf, cf)); break

rows = []
for i, (mf, cf) in enumerate(pairs):
    tag = f"{os.path.basename(os.path.dirname(mf))}/{os.path.basename(mf)[:8]}"
    mr_sitk = sitk.ReadImage(mf)
    ct_r = sitk.Resample(sitk.ReadImage(cf), mr_sitk, sitk.Transform(), sitk.sitkLinear, -1000., sitk.sitkFloat32)
    hu = sitk.GetArrayFromImage(ct_r).astype(np.float32)                 # (z,y,x) real CT HU
    dens_real = hu_to_density(hu, HU).astype(np.float32)
    dens_old, _ = synth_density(mr_sitk, clf_p, net_p, DEV, density_direct=dd, hu_anchors=HU)
    err = dens_old - dens_real                                           # signed density error
    body = hu > -500
    # (1) per-class signed + abs
    r = {"tag": tag}
    for nm, lo, hi in CLASSES:
        m = (hu >= lo) & (hi > hu) & body if nm != "air" else (hu >= lo) & (hi > hu)
        if m.sum() > 100:
            r[f"{nm}_signed"] = float(err[m].mean()); r[f"{nm}_abs"] = float(np.abs(err[m]).mean())
    # (2) boundary vs interior: class-label map, its morphological gradient = boundary voxels
    lab = np.searchsorted([-500, -150, 150], hu).astype(np.int16)
    bnd = ndimage.grey_dilation(lab, size=(3,3,3)) != ndimage.grey_erosion(lab, size=(3,3,3))
    bnd &= body; inter = body & ~bnd
    r["bnd_abs"] = float(np.abs(err[bnd]).mean()); r["inter_abs"] = float(np.abs(err[inter]).mean())
    r["bnd_frac_of_err"] = float(np.abs(err[bnd]).sum() / max(np.abs(err[body]).sum(), 1e-6))
    r["bnd_vox_frac"] = float(bnd.sum() / max(body.sum(), 1))
    # (3) overall + WEPL (range proxy)
    r["body_abs"] = float(np.abs(err[body]).mean())
    r["WEPL"] = float(np.abs(dens_old.sum(0) - dens_real.sum(0))[body.any(0)].mean())
    rows.append(r)
    print(f"[{i}] {tag:22s} body {r['body_abs']:.4f} | bone {r.get('bone_abs',float('nan')):.3f}(s{r.get('bone_signed',0):+.3f}) "
          f"lung {r.get('lung_abs',float('nan')):.3f} | bnd {r['bnd_abs']:.3f} vs inter {r['inter_abs']:.3f} "
          f"({r['bnd_frac_of_err']*100:.0f}% of err in {r['bnd_vox_frac']*100:.0f}% vox) | WEPL {r['WEPL']:.2f}", flush=True)

def m(k): return float(np.nanmean([r[k] for r in rows if k in r]))
print("\n=== MEAN over", len(rows), "test pairs (PRODUCTION sCT error vs real CT) ===")
print(f"  per-class abs:  air {m('air_abs'):.4f}  lung {m('lung_abs'):.4f}  soft {m('soft_abs'):.4f}  bone {m('bone_abs'):.4f}")
print(f"  per-class signed(bias): air {m('air_signed'):+.4f} lung {m('lung_signed'):+.4f} soft {m('soft_signed'):+.4f} bone {m('bone_signed'):+.4f}")
print(f"  BOUNDARY vs INTERIOR:   bnd_abs {m('bnd_abs'):.4f}  inter_abs {m('inter_abs'):.4f}  ratio {m('bnd_abs')/max(m('inter_abs'),1e-6):.2f}x")
print(f"    -> {m('bnd_frac_of_err')*100:.0f}% of total abs-error sits in the {m('bnd_vox_frac')*100:.0f}% of voxels at class boundaries")
print(f"  per-patient body_abs spread: {[round(r['body_abs'],3) for r in rows]}  (variance => OOD signal)")
print(f"  per-patient WEPL spread:     {[round(r['WEPL'],2) for r in rows]}")
