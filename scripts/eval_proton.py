"""Per-beamlet gamma eval for proton-CT models (Task3) — the challenge's scoring tier is the
beamlet. For a spread of validation beamlets, run the model on the cached crop, un-scale, and
score local gamma 1%/1mm + 3%/3mm vs the cached GT (the npz `dose` IS the GT crop). NEW file.

Works for both arms: no-prior (in_ch4) and with-prior (in_ch5, needs prior_dir). Normalisation
matches ProtonDoseDataset exactly. Grid spacing (z,y,x) = (3,1,1) mm.
    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/eval_proton.py \
        --config configs/experiments/proton_ct_noprior.yaml --ckpt <run>/best.pt --label noprior
"""
from __future__ import annotations
import argparse, json, sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statistics as st
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from doserad.model.unet3d import DoseUNet3D
from doserad.eval.gamma import gamma_pass
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE, _P_CH_SCALE_PRIOR

SPACING = (3.0, 1.0, 1.0)   # (z,y,x) mm — proton dose grid


@torch.no_grad()
def _predict(net, inp, dev):
    x = torch.from_numpy(inp[None]).to(dev)
    _, _, Z, Y, X = x.shape
    pz, py, px = (-Z) % 16, (-Y) % 16, (-X) % 16
    if pz or py or px:
        x = F.pad(x, (0, px, 0, py, 0, pz))
    with torch.autocast("cuda", enabled=(dev != "cpu")):
        y = net(x, torch.zeros(1, dtype=torch.long, device=dev))
    return (y[0, 0, :Z, :Y, :X].float() / PROTON_DOSE_SCALE).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-per-patient", type=int, default=40)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(cfg["cache_dir"]); prior = Path(cfg["prior_dir"]) if cfg.get("prior_dir") else None
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]

    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))

    rows = []
    for pid in val:
        d = cache / pid
        files = sorted(f for f in d.glob("B*_R*_L*.npz") if ".tmp" not in f.name)
        if not files:
            continue
        idx = np.linspace(0, len(files) - 1, min(a.n_per_patient, len(files))).astype(int)
        g1s, g3s = [], []
        for j in idx:
            f = files[j]; z = np.load(f)
            ch = z["channels"].astype(np.float32); gt = z["dose"].astype(np.float32)
            if prior is not None:
                pf = prior / pid / f.name
                if not pf.exists():
                    continue
                pb = np.load(pf)["pb_prior"].astype(np.float32)
                inp = np.stack([ch[0], ch[1], pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) / _P_CH_SCALE_PRIOR[:, None, None, None]
            else:
                inp = ch / _P_CH_SCALE[:, None, None, None]
            pred = _predict(net, inp.astype(np.float32), dev)
            rx = float(gt.max())
            if rx <= 0:
                continue
            g1s.append(gamma_pass(pred, gt, SPACING, rx=rx, dose_pct=1.0, dta_mm=1.0))
            g3s.append(gamma_pass(pred, gt, SPACING, rx=rx, dose_pct=3.0, dta_mm=3.0))
        site = "lung" if "THB" in pid else "abdomen"
        rows.append({"patient": pid, "site": site, "n": len(g1s),
                     "g1": float(np.mean(g1s)), "g3": float(np.mean(g3s))})
        print(f"  {pid} ({site}, n={len(g1s)}): beamlet γ1/1 {np.mean(g1s)*100:.1f}%  γ3/3 {np.mean(g3s)*100:.1f}%", flush=True)

    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"proton_eval_{a.label}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    g = lambda k, s: st.mean([r[k] for r in rows if (s is None or r["site"] == s)])
    print(f"\n{a.label}: beamlet γ1/1 ALL {g('g1',None)*100:.1f} (abd {g('g1','abdomen')*100:.1f} / lung {g('g1','lung')*100:.1f}) | "
          f"γ3/3 ALL {g('g3',None)*100:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
