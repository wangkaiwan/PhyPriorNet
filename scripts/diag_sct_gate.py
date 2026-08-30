"""Gate baseline: current 2mm synth, sCT-vs-realCT WEPL on the 16 fold_0 held-out proton val patients.
The 1x1x3 synth must beat this WEPL to justify Stage B."""
import os, sys, json, numpy as np, SimpleITK as sitk
os.environ.setdefault("DOSERAD_CONFIG","configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS","/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF","/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE","/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from container.proton_mri import app
from container.mri_synth import synth_density
from doserad.physics.density import hu_to_density
app.load_models()
clf=app._STATE["clf"]; synth=app._STATE["net"]; dd=app._STATE["density_direct"]; HU=app._STATE["machine"].hu_anchors; DEV=app.DEV
val=json.load(open("/home/kaiwang/doserad2026_workdir/splits_final.json"))["fold_0"]["val"]
base="/data/kwang/DoseRad2026_raw/proton/training"
rows=[]
for pid in val:
    mp,cp=f"{base}/{pid}/image/mr.mha", f"{base}/{pid}/image/ct.mha"
    if not (os.path.exists(mp) and os.path.exists(cp)): continue
    mr=sitk.ReadImage(mp); ct=sitk.ReadImage(cp)
    dens,_=synth_density(mr,clf,synth,DEV,dd)
    im=sitk.GetImageFromArray(dens); im.CopyInformation(mr)
    dr=sitk.Resample(im,ct,sitk.Transform(),sitk.sitkLinear,0.,sitk.sitkFloat32)
    ds=sitk.GetArrayFromImage(dr).astype(np.float32); hu=sitk.GetArrayFromImage(ct).astype(np.float32)
    dc=hu_to_density(hu,HU).astype(np.float32); body=hu>-500; ad=np.abs(ds-dc)
    rows.append({"dens":float(ad[body].mean()),"bone":float(ad[hu>=150].mean()),
                 "WEPL":float(np.abs(ds.sum(0)-dc.sum(0))[(hu>-500).any(0)].mean())})
def m(k): return float(np.mean([r[k] for r in rows]))
print(f"\n=== GATE BASELINE: 2mm synth on {len(rows)} held-out val ===")
for k in ["dens","bone","WEPL"]: print(f"  {k:6s} {m(k):.4f}")
print(">>> the 1x1x3 synth must beat WEPL", round(m("WEPL"),3))
