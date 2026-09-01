"""Plan-γ eval for the MRI→dose models (exp1/exp2). For each val patient, run the model on every
cached CP crop (building [MRI,(sCT),dist,source,open_mask] from the SAME cache + MR/sCT volumes),
accumulate into the full plan dose, and score plan-level local γ1%/1mm vs the GT plan — directly
comparable to the sCT→dose table (v4 best 82.6). NEW file; CT pipeline untouched."""
from __future__ import annotations
import argparse, csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statistics as st
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from doserad.io.mha import load_mha
from doserad.model.unet3d import DoseUNet3D
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.gamma import gamma_array, gamma_pass
from doserad.eval.plan_predict import val_patients_with_cache
from doserad.data.mri_dose_dataset import DOSE_SCALE, CT_MIN, CT_MAX
from doserad.data.dataset import _CH_SCALE, _NAIVE_SCALE
from doserad.physics.naive_dose import compute_naive_dose

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"


def _starts(n, patch, step):
    if n <= patch:
        return [0]
    s = list(range(0, n - patch, step)) + [n - patch]
    return sorted(set(s))


@torch.no_grad()
def _sliding(net, x, m, patch=128, step=96, budget=12_000_000):
    """Inference for one CP crop. FAST PATH: bbox crops are small (~160^3), so do ONE full-crop
    forward (the net is fully-convolutional) padded to a mult of 16 — ~8x fewer ops than tiling
    and frees the GPU from idling on tiny tiles (this is what made the sCT route fast). Falls
    back to 128^3 sliding tiles only if the crop exceeds the voxel budget (won't happen for these
    bbox crops, but kept for safety)."""
    _, _, Z, Y, X = x.shape
    if Z * Y * X <= budget:
        pz, py, px = (-Z) % 16, (-Y) % 16, (-X) % 16
        w = F.pad(x, (0, px, 0, py, 0, pz)) if (pz or py or px) else x
        return net(w, m)[0, 0, :Z, :Y, :X].float().cpu().numpy()
    out = torch.zeros((Z, Y, X), device=x.device)
    cnt = torch.zeros((Z, Y, X), device=x.device)
    for z0 in _starts(Z, patch, step):
        for y0 in _starts(Y, patch, step):
            for x0 in _starts(X, patch, step):
                zz, yy, xx = min(patch, Z), min(patch, Y), min(patch, X)
                w = x[:, :, z0:z0 + zz, y0:y0 + yy, x0:x0 + xx]
                pz, py, px = (-zz) % 16, (-yy) % 16, (-xx) % 16
                if pz or py or px:
                    w = F.pad(w, (0, px, 0, py, 0, pz))
                o = net(w, m)[0, 0, :zz, :yy, :xx]
                out[z0:z0 + zz, y0:y0 + yy, x0:x0 + xx] += o.float()
                cnt[z0:z0 + zz, y0:y0 + yy, x0:x0 + xx] += 1
    return (out / cnt.clamp(min=1)).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-patients", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mode = cfg["mode"]; sct_dir = Path(cfg["sct_dir"]) if cfg.get("sct_dir") else None
    sct_phys_dir = Path(cfg["sct_phys_dir"]) if cfg.get("sct_phys_dir") else None  # exp3
    cache = Path(cfg["cache_dir"])
    _SCT_IMG = ("mri_sct", "mri_sct_phys", "mri_sct_phys_naive")

    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev)
    net.load_state_dict(sd.get("ema", sd.get("net")))
    m = torch.tensor([0], device=dev)

    pids = val_patients_with_cache(cfg)
    if a.max_patients:
        pids = pids[:a.max_patients]
    viz_dir = Path(cfg["run_root"]) / cfg["exp_name"] / "viz"
    rows = []
    for pid in pids:
        mr = load_mha(Path(ROOT) / pid / "image" / "mr.mha")
        a_mr = mr.array.astype(np.float32)
        lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
        mr01 = np.clip((a_mr - lo) / (max(hi - lo, 1.0)), 0, 1).astype(np.float32)
        sct01 = None
        if mode in _SCT_IMG:
            sct_hu = load_mha(sct_dir / pid / "sCT.mha").array.astype(np.float32)
            sct01 = np.clip((sct_hu - CT_MIN) / (CT_MAX - CT_MIN), 0, 1).astype(np.float32)
        full_shape = a_mr.shape

        # Synchronous per-CP loop (NO DataLoader workers: the parent has CUDA initialised,
        # so forking workers dead-locks — "cannot re-init CUDA in forked subprocess"). The
        # single full-crop forward in _sliding is the real speedup (~0.36s/CP -> ~3.3 min/pt).
        cp_files = sorted((cache / pid).glob("*.npz"))
        import time as _t; _t0 = _t.time()
        print(f"[{pid}] inferring {len(cp_files)} CPs ...", flush=True)
        pred_cps, gt_cps = [], []
        for f in cp_files:
            z = np.load(f)
            ch = z["channels"].astype(np.float32); bb = [int(v) for v in z["bbox"]]
            z0, z1, y0, y1, x0, x1 = bb
            sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
            if mode in ("mri_sct_phys", "mri_sct_phys_naive"):
                # exp3: v13 physics channels recomputed from sCT (density, rdepth, fluence) + geometry
                sp = np.load(sct_phys_dir / pid / f.name)["sct_phys"].astype(np.float32)
                d_s, rd_s, fl_s = sp[0], sp[1], sp[2]
                chans = [mr01[sl], sct01[sl],
                         d_s / float(_CH_SCALE[0]), rd_s / float(_CH_SCALE[1]), fl_s / float(_CH_SCALE[2]),
                         ch[3] / float(_CH_SCALE[3]), ch[4] / float(_CH_SCALE[4])]
                if mode == "mri_sct_phys_naive":
                    full5 = np.stack([d_s, rd_s, fl_s, ch[3], ch[4]], 0)
                    chans.append(compute_naive_dose(full5).astype(np.float32) / float(_NAIVE_SCALE))
            else:
                chans = [mr01[sl]]
                if mode == "mri_sct":
                    chans.append(sct01[sl])
                chans += [ch[3] / float(_CH_SCALE[3]), ch[4] / float(_CH_SCALE[4]), (ch[2] > 0).astype(np.float32)]
            inp = np.stack(chans, 0).astype(np.float32)
            x = torch.from_numpy(inp[None]).to(dev)
            with torch.autocast("cuda", enabled=(dev == "cuda")):
                d = _sliding(net, x, m)
            pred_cps.append((d / DOSE_SCALE, tuple(bb)))
            gt_cps.append((z["dose"].astype(np.float32), tuple(bb)))

        plan_pred = accumulate_plan(pred_cps, full_shape)
        plan_gt = accumulate_plan(gt_cps, full_shape)
        rx = float(plan_gt.max())
        zz, yy, xx = np.where(plan_gt >= 0.05 * rx); mgn = 4
        sl = (slice(max(int(zz.min()) - mgn, 0), int(zz.max()) + mgn + 1),
              slice(max(int(yy.min()) - mgn, 0), int(yy.max()) + mgn + 1),
              slice(max(int(xx.min()) - mgn, 0), int(xx.max()) + mgn + 1))
        pc, gc = plan_pred[sl], plan_gt[sl]
        g1c, g1m = gamma_array(pc, gc, mr.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        ov = float((g1c[g1m] <= 1.0).mean()) if g1m.any() else float("nan")
        g3 = gamma_pass(pc, gc, mr.spacing, rx, dose_pct=3.0, dta_mm=3.0)
        # per-patient viz — SAME 6-panel as v13 final_eval (render_plan_figure). Embed the
        # cropped gamma back into the full grid (like eval_sct_route); anatomy backdrop = MR
        # (the model's actual input modality).
        try:
            from doserad.eval.viz import render_plan_figure
            g1_full = np.full(full_shape, np.inf, np.float64); g1_full[sl] = g1c
            mask_full = np.zeros(full_shape, bool); mask_full[sl] = g1m
            viz_dir.mkdir(parents=True, exist_ok=True)
            render_plan_figure(patient=pid, ctarr=a_mr, sp=mr.spacing, gt=plan_gt,
                               pred=plan_pred, g1=g1_full, mask=mask_full, rx=rx,
                               out=str(viz_dir / f"viz_{pid}_{a.label}.png"))
            print(f"  [viz] {pid} -> {viz_dir / f'viz_{pid}_{a.label}.png'}", flush=True)
        except Exception as e:  # noqa: BLE001  (never let viz break the eval)
            print(f"  [viz] {pid} skip ({e})", flush=True)
        site = "lung" if "THB" in pid else "abdomen"
        rows.append({"patient": pid, "site": site, "plan_g1": ov, "plan_g3": g3,
                     "strat_mae": stratified_mae(plan_pred, plan_gt, rx)})
        print(f"  {pid} ({site}): PLAN g1 {ov*100:.1f}%  g3 {g3*100:.1f}%  ({_t.time()-_t0:.0f}s)", flush=True)

    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"dose_mri_{a.label}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    g = lambda s: st.mean([r["plan_g1"] for r in rows if (s is None or r["site"] == s)])
    print(f"\n{a.label}: ALL {g(None)*100:.1f}  abd {g('abdomen')*100:.1f}  lung {g('lung')*100:.1f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
