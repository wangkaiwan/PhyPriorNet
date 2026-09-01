"""sCT HU-MAE of an E2E photon-MRI model's synth, on the SAME val patients the refiner uses, so we can
compare the shipped/final model's sCT quality against the standalone refiner on equal footing. Uses
the exact in-container synth path (mri_synth.synth_density: MR -> clf coarse -> E2E.synth -> sCT01),
with the OLD clf (the front-end that E2E was trained on). Masks match train_sct_refiner.validate.

  CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/sct_mae_e2e.py \
    --cfg configs/experiments/all75/m24S2_p4_mmB.yaml \
    --ckpt <E2E state.pt> --clf /data/kwang/sct_classify_runs/clf_whole/best.pt
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np, torch, yaml, SimpleITK as sitk
from container.mri_synth import synth_density, load_classifier, CT_LO, CT_HI

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--clf", default="/data/kwang/sct_classify_runs/clf_whole/best.pt")
ap.add_argument("--data", default="/home/kaiwang/doserad2026_workdir/sct_data_2mm_samefield.json")
ap.add_argument("--tag", default="E2E")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

from train_dose_e2e import E2E
cfg = yaml.safe_load(open(a.cfg))
net = E2E(cfg).to(dev).eval()
sd = torch.load(a.ckpt, map_location=dev)
net.load_state_dict(sd.get("ema", sd.get("model")))
clf = load_classifier(a.clf, dev)

D = json.load(open(a.data))
val = D["val"]
print(f"[{a.tag}] sCT HU-MAE over {len(val)} val patients (same as refiner val)", flush=True)
allm, bonem, lungm = [], [], []
for it in val:
    mr_sitk = sitk.ReadImage(it["mr"])
    _, sct01 = synth_density(mr_sitk, clf, net, dev, density_direct=True)    # only need sct01; True avoids hu_anchors
    sct_hu = (sct01.float().clamp(0, 1) * (CT_HI - CT_LO) + CT_LO).cpu().numpy()
    ct = sitk.GetArrayFromImage(sitk.ReadImage(it["ct"])).astype(np.float32)
    if ct.shape != sct_hu.shape:                       # guard grid mismatch
        print(f"  WARN {it['pid']} shape {ct.shape} vs {sct_hu.shape} — skip", flush=True); continue
    body = ct > -500; e = np.abs(sct_hu - ct)
    allm.append(e[body].mean())
    if (ct > 200).any(): bonem.append(e[(ct > 200) & body].mean())
    if ((ct < -300) & body).any(): lungm.append(e[(ct < -300) & body].mean())
    print(f"  {it['pid']}: all {e[body].mean():.1f} bone {e[(ct>200)&body].mean():.1f} "
          f"lung {e[(ct<-300)&body].mean():.1f}", flush=True)
print(f"[{a.tag}] MEAN sCT HU-MAE: all {np.mean(allm):.1f} | bone {np.mean(bonem):.1f} | "
      f"lung {np.mean(lungm):.1f}", flush=True)
