"""Per-patient PLAN gamma eval for proton-CT (same method as photon final_eval / eval_sct_route):
accumulate ALL beamlets of a patient into the full-volume plan dose, score plan-level local
gamma 1%/1mm + 3%/3mm vs the GT plan, and render the v13 6-panel viz. NOT sampled — every beamlet
of the patient is predicted and summed (the v13-comparable metric).

Submission tier stays per-beamlet (organizer); this plan view is our INTERNAL model-comparison metric.
    CUDA_VISIBLE_DEVICES=0 conda run -n doserad python scripts/eval_proton_plan.py \
        --config configs/experiments/proton_ct_noprior.yaml --ckpt <run>/best.pt --label noprior
"""
from __future__ import annotations
import argparse, json, sys, os, csv, time
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
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE, _P_CH_SCALE_PRIOR

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
SPACING_ZYX = None   # taken from CT


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
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--save-pred", default=None, help="dir to save plan pred/GT npz per patient (for figs)")
    ap.add_argument("--gpu-pb", action="store_true",
                    help="compute pb_prior on-the-fly via the GPU PB engine (DEPLOYMENT path) "
                         "instead of reading the pyRadPlan cache — deployment-consistency check")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(cfg["cache_dir"]); prior = Path(cfg["prior_dir"]) if cfg.get("prior_dir") else None
    wepl_dir = Path(cfg["wepl_dir"]) if cfg.get("wepl_dir") else None  # corrected ray-march WEPL (ch1) override — MUST match training
    val = json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]
    pm = hu_anchors = None
    if a.gpu_pb:
        from doserad.physics.proton_pb_gpu import ProtonMachineData, proton_pb_dose_gpu
        from doserad.physics.density import hu_to_density
        from doserad.physics.machine import load_photon_machine
        pm = ProtonMachineData(device=dev)
        hu_anchors = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json").hu_anchors
        prior = None  # override: use the GPU engine, not the cache
        print("[gpu-pb] pb_prior computed on-the-fly via proton_pb_gpu (deployment path)", flush=True)
    if a.max_patients:
        val = val[:a.max_patients]
    viz_dir = Path(cfg["run_root"]) / cfg["exp_name"] / "viz_plan"

    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev); net.load_state_dict(sd.get("ema", sd.get("model")))

    rows = []
    for pid in val:
        t0 = time.time()
        ct = load_mha(Path(ROOT) / pid / "image" / "ct.mha")
        full_shape = ct.array.shape
        files = sorted(f for f in (cache / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
        dens_pat = rays = None
        if a.gpu_pb:
            from doserad.physics.density import hu_to_density
            from doserad.physics.proton_pb_gpu import proton_pb_dose_gpu
            dens_pat = hu_to_density(ct.array, hu_anchors).astype(np.float32)   # once per patient
            plan = json.load(open(Path(ROOT) / pid / f"{pid}.json"))
            rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]): (r["ray_source"], r["ray_target"], bl["energy"])
                    for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
        pred_cps, gt_cps = [], []
        for f in files:
            z = np.load(f); ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
            if wepl_dir is not None:  # corrected ray-march WEPL override (ch1) — keep eval == training
                ch[1] = np.load(wepl_dir / pid / f.name)["wepl"].astype(np.float32)
            if a.gpu_pb:
                b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
                src, tgt, e = rays[(b, r, l)]
                pb = proton_pb_dose_gpu(ct, src, tgt, e, out_bbox=bb, machine=pm,
                                        density_override=dens_pat, device=dev).astype(np.float32)
                inp = np.stack([ch[0], ch[1], pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) / _P_CH_SCALE_PRIOR[:, None, None, None]
            elif prior is not None:
                pb = np.load(prior / pid / f.name)["pb_prior"].astype(np.float32)
                inp = np.stack([ch[0], ch[1], pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) / _P_CH_SCALE_PRIOR[:, None, None, None]
            else:
                inp = ch / _P_CH_SCALE[:, None, None, None]
            pred_cps.append((_predict(net, inp.astype(np.float32), dev), bb))
            gt_cps.append((z["dose"].astype(np.float32), bb))
        # Level-1 per-beam masked MAE: mean|pred-gt| over voxels >=10% beam-max GT, / beam-max GT
        bmaes = []
        for (pr, _), (gt, _) in zip(pred_cps, gt_cps):
            gm = gt.max()
            if gm <= 0: continue
            m = gt >= 0.1 * gm
            if m.any(): bmaes.append(float(np.abs(pr[m] - gt[m]).mean() / gm))
        beam_mae = float(np.mean(bmaes)) if bmaes else float("nan")
        plan_pred = accumulate_plan(pred_cps, full_shape)
        plan_gt = accumulate_plan(gt_cps, full_shape)
        rx = float(plan_gt.max())
        zz, yy, xx = np.where(plan_gt >= 0.05 * rx); mgn = 4
        sl = (slice(max(int(zz.min()) - mgn, 0), int(zz.max()) + mgn + 1),
              slice(max(int(yy.min()) - mgn, 0), int(yy.max()) + mgn + 1),
              slice(max(int(xx.min()) - mgn, 0), int(xx.max()) + mgn + 1))
        pc, gc = plan_pred[sl], plan_gt[sl]
        g1c, g1m = gamma_array(pc, gc, ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
        ov = float((g1c[g1m] <= 1.0).mean()) if g1m.any() else float("nan")
        g3 = gamma_pass(pc, gc, ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
        # viz (v13 6-panel), embed cropped gamma back to full grid
        try:
            from doserad.eval.viz import render_plan_figure
            g1f = np.full(full_shape, np.inf, np.float64); g1f[sl] = g1c
            mf = np.zeros(full_shape, bool); mf[sl] = g1m
            viz_dir.mkdir(parents=True, exist_ok=True)
            render_plan_figure(patient=pid, ctarr=ct.array, sp=ct.spacing, gt=plan_gt, pred=plan_pred,
                               g1=g1f, mask=mf, rx=rx, out=str(viz_dir / f"viz_{pid}_{a.label}.png"))
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] {pid} skip ({e})", flush=True)
        if a.save_pred:
            sp = Path(a.save_pred); sp.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp / f"{pid}.npz", pred=plan_pred.astype(np.float32),
                                gt=plan_gt.astype(np.float32), ct=ct.array.astype(np.float32),
                                spacing=np.asarray(ct.spacing, np.float32), rx=np.float32(rx))
        site = "lung" if "THB" in pid else "abdomen"
        rows.append({"patient": pid, "site": site, "plan_g1": ov, "plan_g3": g3, "beam_mae": beam_mae,
                     "strat_mae": stratified_mae(plan_pred, plan_gt, rx)})
        print(f"  {pid} ({site}): PLAN γ1/1 {ov*100:.1f}%  γ3/3 {g3*100:.1f}%  [{len(files)} beamlets, {time.time()-t0:.0f}s]", flush=True)

    out = Path("/home/kaiwang/doserad2026_workdir/runs") / f"proton_plan_{a.label}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    g = lambda s: st.mean([r["plan_g1"] for r in rows if (s is None or r["site"] == s)])
    print(f"\n{a.label}: PLAN γ1/1 ALL {g(None)*100:.1f} (abd {g('abdomen')*100:.1f} / lung {g('lung')*100:.1f}) "
          f"| γ3/3 ALL {st.mean([r['plan_g3'] for r in rows])*100:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
