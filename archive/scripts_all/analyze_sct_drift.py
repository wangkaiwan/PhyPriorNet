"""Quantify how the end-to-end dose-aware training moved the sCT away from the original v4 sCT.
For each val patient: sCT_e2e (the trained synth on MRI) vs sCT_v4 (original) vs real CT.
Reports drift |sCT_e2e - sCT_v4| (overall + per tissue band) and whether HU fidelity vs CT changed,
and whether the drift concentrates in dose-sensitive low-density (lung/air) regions.
    conda run -n doserad python scripts/analyze_sct_drift.py --config <cfg> --ckpt <e2e state.pt>
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
import yaml
from doserad.io.mha import load_mha
from train_dose_e2e import E2E, CT_LO, CT_HI

V4 = Path("/data/kwang/sct_eval/v4")
ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); dev = "cuda"
    net = E2E(cfg).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))
    val = [p.name for p in V4.iterdir() if p.is_dir()]
    print(f"{'patient':9s} {'site':4s} {'drift(all)':>10s} {'soft':>7s} {'lung/air':>9s} {'bone':>7s} "
          f"{'HUMAE_v4':>9s} {'HUMAE_e2e':>10s}")
    rows = []
    for pid in sorted(val):
        mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha"); a_mr = mr.array.astype(np.float32)
        lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
        mr01 = torch.from_numpy(np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(dev)
        sct_e2e = (net.sct01(mr01[None, None])[0, 0] * (CT_HI - CT_LO) + CT_LO).cpu().numpy()
        sct_v4 = load_mha(V4 / pid / "sCT.mha").array.astype(np.float32)
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha").array.astype(np.float32)
        body = ct > -500
        drift = np.abs(sct_e2e - sct_v4)
        lung = body & (ct < -300)            # low-density lung/air (dose-sensitive)
        bone = ct > 200
        soft = body & (ct >= -300) & (ct <= 200)
        site = "lung" if "THB" in pid else "abd"
        def m(msk): return float(drift[msk].mean()) if msk.any() else 0.0
        hv = float(np.abs(sct_v4 - ct)[body].mean()); he = float(np.abs(sct_e2e - ct)[body].mean())
        rows.append((pid, site, m(body), m(soft), m(lung), m(bone), hv, he))
        print(f"{pid:9s} {site:4s} {m(body):10.1f} {m(soft):7.1f} {m(lung):9.1f} {m(bone):7.1f} {hv:9.1f} {he:10.1f}")
    import statistics as st
    for lab, f in [("ALL", lambda r: True), ("abd", lambda r: r[1] == "abd"), ("lung", lambda r: r[1] == "lung")]:
        s = [r for r in rows if f(r)]
        print(f"[{lab}] drift all {st.mean(x[2] for x in s):.1f} | lung/air {st.mean(x[4] for x in s):.1f} | "
              f"HUMAE v4 {st.mean(x[6] for x in s):.1f} -> e2e {st.mean(x[7] for x in s):.1f} HU")


if __name__ == "__main__":
    main()
