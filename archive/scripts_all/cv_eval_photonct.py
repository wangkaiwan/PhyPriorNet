"""Fast Photon-CT plan eval for CV: build plan per val patient (cropped gamma), record all paper metrics.
Usage: cv_eval_photonct.py --config CFG --ckpt CKPT --out OUT.csv"""
import sys, argparse, csv, yaml; sys.path.insert(0, '.')
import numpy as np, torch
from pathlib import Path
from doserad.model.unet3d import DoseUNet3D
from doserad.data.dataset import DOSE_SCALE, normalize_channels
from doserad.eval.plan_predict import ROOT, predict_cp
from doserad.eval.plan_agg import accumulate_plan, stratified_mae
from doserad.eval.gamma import gamma_array, gamma_pass
from doserad.beam.parse import load_photon_plan
from doserad.io.mha import load_mha

ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); dev = 'cuda'
import json
val = json.load(open(cfg['splits']))[f"fold_{cfg['fold']}"]['val']
net = DoseUNet3D(in_ch=cfg.get('in_ch', 5), base=cfg['base_ch'], levels=cfg['levels'], bottleneck=cfg.get('bottleneck', 'plain')).to(dev).eval()
net.load_state_dict(torch.load(a.ckpt, map_location=dev)['ema'])
cache = Path(cfg['cache_dir']); rows = []; worst = None
for pid in val:
    plan = load_photon_plan(Path(ROOT) / pid / f'{pid}.json'); ct = load_mha(Path(ROOT) / pid / 'image' / 'ct.mha'); full = ct.array.shape
    pc, gc, bmaes = [], [], []
    for beam in plan.beams:
        for cp in beam.control_points:
            f = cache / pid / f'{beam.beam_idx}_{cp.cp_idx:03d}.npz'
            if not f.exists(): continue
            z = np.load(f); ch = normalize_channels(z['channels'], add_naive=cfg.get('add_naive', False), naive_skin_gate=cfg.get('naive_skin_gate')).astype(np.float32)
            pr = predict_cp(net, ch, dev) / DOSE_SCALE; gt = z['dose'].astype(np.float32); bb = tuple(int(v) for v in z['bbox'])
            pc.append((pr, bb)); gc.append((gt, bb))
            gm = gt.max()
            if gm > 0:
                m = gt >= 0.1 * gm
                if m.any(): bmaes.append(float(np.abs(pr[m] - gt[m]).mean() / gm))
    pp = accumulate_plan(pc, full); gt = accumulate_plan(gc, full); rx = float(gt.max())
    zz, yy, xx = np.where(gt >= 0.05 * rx); mg = 4
    sl = (slice(max(int(zz.min())-mg, 0), int(zz.max())+mg+1), slice(max(int(yy.min())-mg, 0), int(yy.max())+mg+1), slice(max(int(xx.min())-mg, 0), int(xx.max())+mg+1))
    g1c, g1m = gamma_array(pp[sl], gt[sl], ct.spacing, rx, dose_pct=1.0, dta_mm=1.0)
    g1 = float((g1c[g1m] <= 1).mean()) if g1m.any() else float('nan')
    g2 = gamma_pass(pp[sl], gt[sl], ct.spacing, rx, dose_pct=2.0, dta_mm=2.0)
    g3 = gamma_pass(pp[sl], gt[sl], ct.spacing, rx, dose_pct=3.0, dta_mm=3.0)
    site = 'lung' if 'THB' in pid else 'abdomen'
    rows.append({'patient': pid, 'site': site, 'plan_g1': g1, 'plan_g2': g2, 'plan_g3': g3,
                 'beam_mae': float(np.mean(bmaes)) if bmaes else float('nan'), 'strat_mae': stratified_mae(pp, gt, rx)})
    print(f"  {pid} ({site}): g1 {g1*100:.1f} g2 {g2*100:.1f} g3 {g3*100:.1f} beam_mae {rows[-1]['beam_mae']*100:.2f}%", flush=True)
    if not np.isnan(g1) and (worst is None or g1 < worst[0]):
        g1f = np.full(gt.shape, np.inf); g1f[sl] = g1c; mf = np.zeros(gt.shape, bool); mf[sl] = g1m
        worst = (g1, pid, pp.copy(), gt.copy(), ct.array.copy(), ct.spacing, rx, g1f, mf)
with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import statistics as st
print(f"[{cfg['exp_name']}] mean g1 {st.mean([r['plan_g1'] for r in rows])*100:.1f} -> {a.out}", flush=True)
if worst is not None:
    from doserad.eval.viz import render_plan_figure
    _g, _pid, _pp, _gt, _ct, _sp, _rx, _g1f, _mf = worst
    outp = str(Path(a.out).parent / f"worst_{cfg['exp_name'].replace('cv_','')}_{_pid}.png")
    try:
        render_plan_figure(patient=f"{_pid} (WORST {_g*100:.1f}%)", ctarr=_ct, sp=_sp, gt=_gt, pred=_pp, g1=_g1f, mask=_mf, rx=_rx, out=outp)
        print(f"[worst] {_pid} g1 {_g*100:.1f}% -> {outp}", flush=True)
    except Exception as e:
        print(f"[worst] render skip: {e}", flush=True)
