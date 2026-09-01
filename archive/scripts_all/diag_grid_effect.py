"""Experiment 2 (corrected): the synth works internally at 2mm, but proton MR/CT are native 1x1x3
(train AND test — so NO train/test grid mismatch). Question: does the 2mm resolution bottleneck cost
range/density accuracy? Measure MODEL-INDEPENDENTLY: push the REAL CT through 1x1x3 -> 2mm -> 1x1x3 and
compare to the original. That loss = the ceiling a native-1x1x3 synth could recover."""
import os, glob, numpy as np, SimpleITK as sitk, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doserad.physics.density import hu_to_density
from doserad.io.mha import load_mha
from doserad.physics.machine import load_photon_machine
HU = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json").hu_anchors

def to_grid(img, sp):
    size=[int(round(img.GetSize()[i]*img.GetSpacing()[i]/sp[i])) for i in range(3)]
    g=sitk.Image(size, sitk.sitkFloat32); g.SetOrigin(img.GetOrigin()); g.SetDirection(img.GetDirection()); g.SetSpacing(sp)
    return sitk.Resample(img, g, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)

tr = sorted(glob.glob("/data/kwang/DoseRad2026_raw/proton/training/1*/image"))[:10]
rows=[]
for d in tr:
    cp=f"{d}/ct.mha"
    if not os.path.exists(cp): continue
    ct = sitk.ReadImage(cp)                                   # native 1x1x3
    ct_bottleneck = sitk.Resample(to_grid(ct,(2.,2.,2.)), ct, sitk.Transform(), sitk.sitkLinear, -1000., sitk.sitkFloat32)  # 1x1x3->2mm->1x1x3
    hu0 = sitk.GetArrayFromImage(ct).astype(np.float32); hu2 = sitk.GetArrayFromImage(ct_bottleneck).astype(np.float32)
    d0 = hu_to_density(hu0,HU).astype(np.float32); d2 = hu_to_density(hu2,HU).astype(np.float32)
    body = hu0>-500; ad=np.abs(d0-d2)
    rows.append({"dens_MAE":float(ad[body].mean()), "bone":float(ad[hu0>=150].mean()),
                 "WEPL":float(np.abs(d0.sum(0)-d2.sum(0))[(hu0>-500).any(0)].mean())})
def m(k): return float(np.mean([r[k] for r in rows]))
print(f"\n=== 2mm-bottleneck resolution loss on REAL CT (n={len(rows)}) ===")
print("  (how much the 2mm synth bottleneck loses vs native 1x1x3; = ceiling a 1x1x3 synth could recover)")
for k in ["dens_MAE","bone","WEPL"]:
    print(f"  {k:10s}  loss {m(k):.4f}")
print("\n  compare: sCT's OWN error on training (Exp1) = dens 0.052 / bone 0.143 / WEPL 1.83")
