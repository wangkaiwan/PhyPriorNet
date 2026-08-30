"""Rigorous accuracy check: does torch.compile(dynamic=True) change the SCORED metric (plan
gamma 1%/1mm), not just the dose rel-diff? Runs the exact cv_eval_photonct scoring loop on the
fold-0 val set with the net compiled, and compares per-patient plan gamma against the eager
baseline (/tmp/gl_base.csv). New file — parent CT-dose pipeline untouched.

Run (GPU1): CUDA_VISIBLE_DEVICES=1 conda run -n doserad python -u accel/gamma_compile_check.py
"""
from __future__ import annotations
import sys, csv, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, yaml
from doserad.model.unet3d import DoseUNet3D
from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.plan_predict import ROOT, predict_cp
from doserad.eval.plan_agg import accumulate_plan
from doserad.eval.gamma import gamma_array
from doserad.beam.parse import load_photon_plan
from doserad.io.mha import load_mha

CFG = "configs/experiments/cv/ftg_skinentry_photonct_f0.yaml"
CKPT = "/home/kaiwang/doserad2026_workdir/runs/ftg_skinentry_photonct_f0/state.pt"
BASE_CSV = "/tmp/gl_base.csv"   # eager baseline (same model) from the gammaloss run
dev = "cuda"


def main():
    cfg = yaml.safe_load(open(CFG))
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
    net = DoseUNet3D(in_ch=cfg.get("in_ch", 6), base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "dilated")).to(dev).eval()
    net.load_state_dict(torch.load(CKPT, map_location=dev)["ema"])
    net = torch.compile(net, dynamic=True)   # the ONLY change vs cv_eval_photonct
    cache = Path(cfg["cache_dir"])
    base = {r["patient"]: float(r["plan_g1"]) * 100 for r in csv.DictReader(open(BASE_CSV))}

    print("patient       eager  compiled  delta")
    rows = []
    for pid in val:
        plan = load_photon_plan(Path(ROOT) / pid / f"{pid}.json")
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha"); full = ct.array.shape
        pc, gc = [], []
        for beam in plan.beams:
            for cp in beam.control_points:
                f = cache / pid / f"{beam.beam_idx}_{cp.cp_idx:03d}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                ch = normalize_channels(z["channels"], add_naive=cfg.get("add_naive", False),
                                        naive_skin_gate=cfg.get("naive_skin_gate")).astype(np.float32)
                pr = predict_cp(net, ch, dev) / DOSE_SCALE
                bb = tuple(int(v) for v in z["bbox"])
                pc.append((pr, bb)); gc.append((z["dose"].astype(np.float32), bb))
        pp = accumulate_plan(pc, full); gt = accumulate_plan(gc, full); rx = float(gt.max())
        zz, yy, xx = np.where(gt >= 0.05 * rx); mg = 4
        sl = (slice(max(int(zz.min()) - mg, 0), int(zz.max()) + mg + 1),
              slice(max(int(yy.min()) - mg, 0), int(yy.max()) + mg + 1),
              slice(max(int(xx.min()) - mg, 0), int(xx.max()) + mg + 1))
        g1c, g1m = gamma_array(pp[sl], gt[sl], ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        g1 = float((g1c[g1m] <= 1).mean()) * 100 if g1m.any() else float("nan")
        e = base.get(pid, float("nan"))
        rows.append((pid, e, g1))
        print(f"{pid:12s}  {e:5.1f}   {g1:5.1f}   {g1-e:+.2f}", flush=True)
    eg = np.array([r[1] for r in rows]); cg = np.array([r[2] for r in rows])
    print(f"\nMEAN plan gamma1/1: eager {eg.mean():.2f}  compiled {cg.mean():.2f}  "
          f"delta {cg.mean()-eg.mean():+.3f}")
    print(f"max |per-patient delta|: {np.abs(cg-eg).max():.2f}")


if __name__ == "__main__":
    main()
