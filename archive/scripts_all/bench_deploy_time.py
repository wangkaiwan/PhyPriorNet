"""DEPLOYMENT inference-timing benchmark for all 4 DoseRAD2026 tasks.

Measures, per task, the average per-beam-element and per-patient DEPLOYMENT
inference time. Timed region = channel/prior build + dose-net forward + plan
accumulation ONLY. Excluded: gamma, ground-truth loading for metrics, MC, viz,
and one-time model load. For the two MRI tasks the one-time MR->sCT synthesis is
included in the per-patient time AND reported separately.

Reuses the deployed skin-entry models and the physics helpers from the existing
eval scripts (no physics re-derivation). FP32 default (autocast as in the eval
forwards, i.e. the deployed baseline). GPU-synced timers around every timed
region; one warm-up patient per task (excluded) to skip first-call cuDNN/CUDA
autotune.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n doserad --no-capture-output python -u scripts/bench_deploy_time.py \
    [--task photon_ct,proton_ct,photon_mri,proton_mri] [--warmup 1] [--timed 2] \
    [--out docs/reports/deploy_timing_baseline.md]

Beam element: photon = control point (CP); proton = beamlet.
"""
from __future__ import annotations
import argparse, json, os, sys, time, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root
sys.path.insert(0, _HERE)                    # scripts dir (train_dose_e2e, eval_* modules)
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PHOTON_ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
PROTON_ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
MACHINE = "/data/kwang/DoseRad2026_raw/beam_parameters.json"
WORKDIR = "/home/kaiwang/doserad2026_workdir"
CFG = f"{os.path.dirname(_HERE)}/configs/experiments/cv"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _val_pids(cfg):
    return json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]


def _measure(run_patient, val, n_warmup, n_timed, has_synth=False):
    """run_patient(pid) -> (per_patient_s, n_elems, synth_s). Warm up on the first
    n_warmup pids (discarded), time the next n_timed."""
    pids = [p for p in val][: n_warmup + n_timed]
    for pid in pids[:n_warmup]:
        print(f"    [warmup] {pid} ...", flush=True)
        run_patient(pid)
    recs = []
    for pid in pids[n_warmup: n_warmup + n_timed]:
        dt, n, s = run_patient(pid)
        recs.append(dict(pid=pid, t=dt, n=n, synth=s))
        msg = f"    {pid}: {dt:7.2f}s  {n:5d} elems  {dt / n * 1000:7.2f} ms/elem"
        if has_synth:
            msg += f"  (synth {s:.2f}s)"
        print(msg, flush=True)
    return recs


# ----------------------------------------------------------------------------- photon-CT
def bench_photon_ct(n_warmup, n_timed):
    from doserad.model.unet3d import DoseUNet3D
    from doserad.physics.channels import photon_channels
    from doserad.physics.density import hu_to_density
    from doserad.physics.machine import load_photon_machine
    from doserad.inference.pipeline import _normalize_gpu, _build_coords
    from doserad.beam.parse import load_photon_plan
    from doserad.io.mha import load_mha
    from doserad.data.dataset import DOSE_SCALE

    cfg = yaml.safe_load(open(f"{CFG}/ftg_skinentry_photonct_f0.yaml"))
    ckpt = f"{WORKDIR}/runs/ftg_skinentry_photonct_f0/state.pt"
    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(DEV).eval()
    net.load_state_dict(torch.load(ckpt, map_location=DEV)["ema"])
    machine = load_photon_machine(MACHINE)
    add_naive = bool(cfg.get("add_naive", False))
    factor, infer_batch, max_batch_voxels = 8, 8, 2_500_000

    def run_patient(pid):
        ct = load_mha(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
        plan = load_photon_plan(Path(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
        full_shape = ct.array.shape
        sync(); t0 = time.time()
        density = hu_to_density(ct.array, machine.hu_anchors)
        coords = _build_coords(ct, "cuda" if torch.cuda.is_available() else "cpu")
        plan_dose = np.zeros(full_shape, np.float32)
        buf, mx, ncp = [], [0, 0, 0], 0

        def flush(buf):
            dims = [c.shape[-3:] for c, _ in buf]
            D = max(d[0] for d in dims); H = max(d[1] for d in dims); W = max(d[2] for d in dims)
            D = -(-D // factor) * factor; H = -(-H // factor) * factor; W = -(-W // factor) * factor
            B, C = len(buf), buf[0][0].shape[0]
            xb = torch.zeros((B, C, D, H, W), dtype=torch.float32, device=DEV)
            for i, (c, _) in enumerate(buf):
                d, h, w = c.shape[-3:]; xb[i, :, :d, :h, :w] = c.to(DEV)
            m = torch.zeros(B, dtype=torch.long, device=DEV)
            with torch.no_grad(), torch.autocast("cuda", enabled=(DEV != "cpu")):
                yb = net(xb, m).float()
            yb = (yb / DOSE_SCALE).cpu().numpy()
            for i, (c, bbox) in enumerate(buf):
                d, h, w = c.shape[-3:]; z0, z1, y0, y1, x0, x1 = bbox
                plan_dose[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += yb[i, 0, :d, :h, :w]

        for beam in plan.beams:
            for cp in beam.control_points:
                crop, bbox = photon_channels(
                    image=ct, machine=machine, iso_xyz=beam.iso_center,
                    gantry_deg=cp.gantry_angle,
                    mlc_left=np.asarray(cp.mlc_left_int_mm),
                    mlc_right=np.asarray(cp.mlc_right_int_mm),
                    density_override=density, coords=coords,
                    crop_margin=8, return_tensor=True)
                crop = _normalize_gpu(crop, add_naive)
                d, h, w = crop.shape[-3:]
                nmx = [max(mx[0], d), max(mx[1], h), max(mx[2], w)]
                if buf and ((len(buf) + 1) * nmx[0] * nmx[1] * nmx[2] > max_batch_voxels
                            or len(buf) >= infer_batch):
                    flush(buf); buf = []; mx = [0, 0, 0]; nmx = [d, h, w]
                buf.append((crop, bbox)); mx = nmx; ncp += 1
        if buf:
            flush(buf)
        sync(); return time.time() - t0, ncp, 0.0

    return _measure(run_patient, _val_pids(cfg), n_warmup, n_timed)


# ----------------------------------------------------------------------------- photon-MRI
def bench_photon_mri(n_warmup, n_timed):
    from doserad.io.mha import load_mha
    from doserad.physics.machine import load_photon_machine
    from doserad.physics.geometry import beam_source_pos, beam_basis
    from doserad.beam.parse import load_photon_plan
    from doserad.physics.diff_channels import (hu_to_density_torch, radiological_depth_fast_torch,
                                               fluence_torch, naive_dose_torch)
    from doserad.physics.diff_channels_skinentry import radiological_depth_skinentry_torch
    from doserad.data.dataset import _CH_SCALE, _NAIVE_SCALE, DOSE_SCALE
    from doserad.eval.plan_agg import accumulate_plan
    from train_dose_e2e import E2E, CT_LO, CT_HI, _pad16

    cfg = yaml.safe_load(open(f"{CFG}/se_photonmri_f0.yaml"))
    ckpt = f"{WORKDIR}/runs/se_photonmri_f0/best.pt"
    ROOT = cfg["root_dir"]; machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    anchors = machine.hu_anchors
    skin_entry = bool(cfg.get("naive_skin_gate", False))
    rdepth_fn = radiological_depth_skinentry_torch if skin_entry else radiological_depth_fast_torch
    cache = Path(cfg["cache_dir"]); mod = torch.zeros(1, dtype=torch.long, device=DEV)
    net = E2E(cfg).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV); net.load_state_dict(sd.get("ema", sd.get("model")))
    img_ch = int(cfg["in_ch"]) > 6

    def run_patient(pid):
        mr = load_mha(f"{ROOT}/{pid}/image/mr.mha")
        a_mr = mr.array.astype(np.float32)
        lo, hi = np.percentile(a_mr, 1), np.percentile(a_mr, 99)
        mr01 = torch.from_numpy(np.clip((a_mr - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)).to(DEV)
        sync(); t0 = time.time()
        # --- one-time MR -> sCT -> density synthesis (reported separately) ---
        with torch.no_grad(), torch.autocast("cuda"):
            if cfg.get("coarse_dir"):
                cv = load_mha(f"{cfg['coarse_dir']}/{pid}.nii.gz").array.astype(np.float32)
                co = torch.from_numpy(np.clip((cv - CT_LO) / (CT_HI - CT_LO), 0, 1).astype(np.float32)).to(DEV)
                sct01 = net.sct01(torch.stack([mr01, co], 0)[None])[0, 0]
            else:
                sct01 = net.sct01(mr01[None, None])[0, 0]
            sct_hu = sct01 * (CT_HI - CT_LO) + CT_LO
            density = hu_to_density_torch(sct_hu, anchors).float()
        sync(); synth_t = time.time() - t0
        # geometry (per CP source/basis)
        pl = load_photon_plan(Path(f"{ROOT}/{pid}/{pid}.json")); geo = {}
        for b in pl.beams:
            for cp in b.control_points:
                iso = np.asarray(b.iso_center, np.float64)
                geo[f"{b.beam_idx}_{cp.cp_idx:03d}"] = (iso, beam_source_pos(iso, machine.sad_mm, cp.gantry_angle),
                                                        *beam_basis(cp.gantry_angle))
        pred_cps = []
        for f in sorted((cache / pid).glob("*.npz")):
            z = np.load(f); ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
            z0, z1, y0, y1, x0, x1 = bb; sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
            iso, src, ax, u, v = geo[f.stem]
            with torch.no_grad(), torch.autocast("cuda"):
                rdepth = rdepth_fn(density, mr.spacing, mr.origin, src, ax, u, v, iso, out_bbox=bb)
                cht = torch.from_numpy(ch).to(DEV)
                fl = fluence_torch((cht[2] > 0).float(), rdepth)
                naive = naive_dose_torch(density[sl], rdepth, fl, cht[4], skin_gate=skin_entry)
                chans = [density[sl] / _CH_SCALE[0], rdepth / _CH_SCALE[1], fl / _CH_SCALE[2],
                         cht[3] / _CH_SCALE[3], cht[4] / _CH_SCALE[4], naive / _NAIVE_SCALE]
                if img_ch:
                    chans += [mr01[sl], sct01[sl]]
                inp = torch.stack(chans, 0)
                Z, Y, X = inp.shape[-3:]
                inp = F.pad(inp[None], (0, _pad16(X), 0, _pad16(Y), 0, _pad16(Z)))
                pred = net.dose(inp, mod)[0, 0, :Z, :Y, :X].float() / DOSE_SCALE
            pred_cps.append((pred.cpu().numpy(), bb))
        _ = accumulate_plan(pred_cps, a_mr.shape)
        sync(); return time.time() - t0, len(pred_cps), synth_t

    return _measure(run_patient, _val_pids(cfg), n_warmup, n_timed, has_synth=True)


# ----------------------------------------------------------------------------- proton (shared)
def _bench_proton(task, n_warmup, n_timed):
    import eval_protonmri_plan as epm
    from doserad.physics.proton_pb_gpu import ProtonMachineData
    from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry
    from doserad.physics.machine import load_photon_machine
    from doserad.physics.density import hu_to_density
    from doserad.model.unet3d import DoseUNet3D
    from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
    from doserad.eval.plan_agg import accumulate_plan
    from doserad.io.mha import load_mha
    from train_dose_e2e import E2E
    import SimpleITK as sitk

    machine = load_photon_machine(MACHINE); hu_anchors = machine.hu_anchors
    pm = ProtonMachineData(device=DEV)
    pb_fn = proton_pb_dose_gpu_skinentry; mask_kw = {}   # both deployed proton models use skin-entry prior

    if task == "proton_ct":
        cfg = yaml.safe_load(open(f"{CFG}/ft_skinentry_protonct_f0.yaml"))
        eng = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                         bottleneck=cfg.get("bottleneck", "plain")).to(DEV).eval()
        sd = torch.load(f"{WORKDIR}/runs/ft_skinentry_protonct_f0/state.pt", map_location=DEV)
        eng.load_state_dict(sd.get("ema", sd.get("model")))
        synth = scfg = None
    else:
        scfg = yaml.safe_load(open(f"{CFG}/se_protonmri_f0.yaml"))
        e2e = E2E(scfg).to(DEV).eval()
        sd = torch.load(f"{WORKDIR}/runs/se_protonmri_f0/best.pt", map_location=DEV)
        e2e.load_state_dict(sd.get("ema", sd.get("model")))
        synth, eng, cfg = e2e, e2e.dose, scfg
    cache_root = Path(cfg["cache_dir"])

    def run_patient(pid):
        ct = load_mha(f"{PROTON_ROOT}/{pid}/image/ct.mha")
        plan_json = f"{PROTON_ROOT}/{pid}/{pid}.json"
        sync(); t0 = time.time(); synth_t = 0.0
        if task == "proton_ct":
            density = hu_to_density(ct.array, hu_anchors).astype(np.float32)
        else:
            ct_sitk = sitk.ReadImage(f"{PROTON_ROOT}/{pid}/image/ct.mha")
            sync(); ts = time.time()
            density, _ = epm._make_sct_density(synth, scfg, pid, ct_sitk, hu_anchors, DEV)
            sync(); synth_t = time.time() - ts
        # per-beamlet: WEPL (ray-march on density) + PB prior (GPU engine) + net forward
        plan = json.load(open(plan_json))
        rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
                (r["ray_source"], r["ray_target"], bl["energy"])
                for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
        files = sorted(f for f in (cache_root / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
        pred_cps = []
        for f in files:
            z = np.load(f); ch = z["channels"].astype(np.float32); bb = tuple(int(v) for v in z["bbox"])
            z0, z1, y0, y1, x0, x1 = bb; sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
            b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
            src, tgt, e = rays[(b, r, l)]
            dens_c = density[sl]
            wepl_c = epm._wepl_on_density(ct, density, src, tgt, bb, pm, DEV)
            pb = pb_fn(ct, src, tgt, e, out_bbox=bb, machine=pm,
                       density_override=density, device=DEV, **mask_kw).astype(np.float32)
            inp = np.stack([dens_c, wepl_c, pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) \
                / _P_CH_SCALE_PRIOR[:, None, None, None]
            pred_cps.append((epm._predict(eng, inp.astype(np.float32), DEV), bb))
        _ = accumulate_plan(pred_cps, ct.array.shape)
        sync(); return time.time() - t0, len(files), synth_t

    return _measure(run_patient, _val_pids(cfg), n_warmup, n_timed, has_synth=(task == "proton_mri"))


def bench_proton_ct(n_warmup, n_timed):  return _bench_proton("proton_ct", n_warmup, n_timed)
def bench_proton_mri(n_warmup, n_timed): return _bench_proton("proton_mri", n_warmup, n_timed)


TASKS = {
    "photon_ct":  ("Photon-CT",  "control point (CP)", bench_photon_ct),
    "proton_ct":  ("Proton-CT",  "beamlet",            bench_proton_ct),
    "photon_mri": ("Photon-MRI", "control point (CP)", bench_photon_mri),
    "proton_mri": ("Proton-MRI", "beamlet",            bench_proton_mri),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="photon_ct,proton_ct,photon_mri,proton_mri")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--out", default=f"{os.path.dirname(_HERE)}/docs/reports/deploy_timing_baseline.md")
    a = ap.parse_args()

    import subprocess
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"GPU(s):\n{gpu}\nCUDA_VISIBLE_DEVICES={vis}  device={DEV}  torch={torch.__version__}", flush=True)

    summary = []
    for tk in a.task.split(","):
        tk = tk.strip()
        if tk not in TASKS:
            print(f"[skip] unknown task {tk}"); continue
        name, elem, fn = TASKS[tk]
        print(f"\n=== {name} (beam element = {elem}) ===", flush=True)
        try:
            recs = fn(a.warmup, a.timed)
            n_elems = st.mean([r["n"] for r in recs])
            per_patient = st.mean([r["t"] for r in recs])
            per_elem_ms = st.mean([r["t"] / r["n"] * 1000 for r in recs])
            synth = st.mean([r["synth"] for r in recs])
            summary.append(dict(task=name, elem=elem, n_elems=n_elems, per_elem_ms=per_elem_ms,
                                per_patient=per_patient, synth=synth, n_timed=len(recs)))
        except Exception as ex:  # noqa: BLE001
            import traceback; traceback.print_exc()
            summary.append(dict(task=name, elem=elem, n_elems=float("nan"), per_elem_ms=float("nan"),
                                per_patient=float("nan"), synth=float("nan"), n_timed=0,
                                error=str(ex)))

    # ---- table ----
    hdr = ("| Task | beam element | avg #elements/patient | per-beam-element (ms) "
           "| per-patient (s) | synth once/patient (s, MRI only) |")
    sep = "|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for s in summary:
        synth = f"{s['synth']:.2f}" if s["synth"] and s["synth"] > 0 else "-"
        lines.append(f"| {s['task']} | {s['elem']} | {s['n_elems']:.0f} | "
                     f"{s['per_elem_ms']:.2f} | {s['per_patient']:.2f} | {synth} |")
    table = "\n".join(lines)
    print("\n" + table, flush=True)

    Path(os.path.dirname(a.out)).mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("# DoseRAD2026 — Deployment Inference-Timing Baseline (FP32)\n\n")
        fh.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}\n")
        fh.write(f"- Hardware: {gpu.splitlines()[int(vis) if vis.isdigit() else 0].strip() if gpu else 'n/a'} "
                 f"(CUDA_VISIBLE_DEVICES={vis})\n")
        fh.write(f"- torch {torch.__version__}, FP32 baseline (pre-acceleration; autocast as in the deployed eval forwards)\n")
        fh.write(f"- Timed patients per task: {a.timed} fold-0 val patients (+{a.warmup} warm-up, excluded)\n")
        fh.write("- Timed region = per-patient channel/prior build + dose-net forward + plan accumulation. "
                 "Excludes model load, gamma, ground-truth load, MC, viz. MRI: MR->sCT synth included in "
                 "per-patient time and reported separately.\n\n")
        fh.write(table + "\n")
    print(f"\n[written] {a.out}", flush=True)


if __name__ == "__main__":
    main()
