"""Stage A (cheap, no retrain): run the existing 2mm-trained synth at NATIVE 1x1x3 (skip the 2mm
downsample) and compare sCT-vs-realCT to the current 2mm-roundtrip. If native inference recovers the
grid loss -> free win (just change container). If worse (2mm net is OOD at 1x1x3) -> need Stage B retrain."""
import os, sys, glob
os.environ.setdefault("DOSERAD_CONFIG", "configs/experiments/all75/all75_r3_protonmri.yaml")
os.environ.setdefault("DOSERAD_WEIGHTS", "/home/kaiwang/doserad2026_workdir/runs/all75_r3ft_mraug_protonmri/state.pt")
os.environ.setdefault("DOSERAD_CLF", "/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt")
os.environ.setdefault("DOSERAD_MACHINE", "/data/kwang/DoseRad2026_raw/beam_parameters.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, SimpleITK as sitk
from container.proton_mri import app
import container.mri_synth as ms
from doserad.physics.density import hu_to_density
app.load_models()
clf = app._STATE["clf"]; synth = app._STATE["net"]; dd = app._STATE["density_direct"]
HU = app._STATE["machine"].hu_anchors; DEV = app.DEV

def err(dens_src_np, mr, ct):
    im = sitk.GetImageFromArray(dens_src_np); im.CopyInformation(mr)
    dr = sitk.Resample(im, ct, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ds = sitk.GetArrayFromImage(dr).astype(np.float32)
    hu = sitk.GetArrayFromImage(ct).astype(np.float32); dc = hu_to_density(hu, HU).astype(np.float32)
    body = hu > -500; ad = np.abs(ds-dc)
    return {"dens":float(ad[body].mean()),"bone":float(ad[hu>=150].mean()),
            "WEPL":float(np.abs(ds.sum(0)-dc.sum(0))[(hu>-500).any(0)].mean())}

tr = sorted(glob.glob("/data/kwang/DoseRad2026_raw/proton/training/1*/image"))[:8]
R2=[]; RN=[]
for d in tr:
    mp,cp=f"{d}/mr.mha",f"{d}/ct.mha"
    if not (os.path.exists(mp) and os.path.exists(cp)): continue
    mr=sitk.ReadImage(mp); ct=sitk.ReadImage(cp)
    ms._SYNTH_SPACING=(2.0,2.0,2.0); d2,_=ms.synth_density(mr,clf,synth,DEV,dd); R2.append(err(d2,mr,ct))   # current 2mm
    try:
        ms._SYNTH_SPACING=(1.0,1.0,3.0); dn,_=ms.synth_density(mr,clf,synth,DEV,dd); RN.append(err(dn,mr,ct)) # native 1x1x3
    except RuntimeError as e:
        print("  native OOM/err on", d.split('/')[-2], str(e)[:60]); RN.append(None)
RN=[r for r in RN if r]
def m(rows,k): return float(np.mean([r[k] for r in rows]))
print(f"\n=== Stage A: 2mm-synth vs native-1x1x3-synth (vs real CT, n2={len(R2)} nN={len(RN)}) ===")
for k in ["dens","bone","WEPL"]:
    print(f"  {k:6s}  2mm-roundtrip {m(R2,k):.4f}   native-1x1x3 {m(RN,k):.4f}   Δ {m(RN,k)-m(R2,k):+.4f}")
