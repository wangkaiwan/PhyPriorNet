"""ACCELERATION prototypes for DoseRAD2026 deployment inference (CT tasks).

Standalone benchmark — does NOT touch production inference/eval code. For
Photon-CT (per-CP) and Proton-CT (per-beamlet) it measures, with the same
GPU-synced timing protocol as scripts/bench_deploy_time.py, the variants:

  seq_amp            batch-1 forward + torch.autocast          <- accuracy REFERENCE
                     (this is what the deployed proton/MRI loops do; the deployed
                      photon-CT path already streams batches of <=8 + autocast)
  seq_fp32           batch-1, autocast OFF (true FP32; cuDNN TF32-conv defaults)
  batch{4,8,16}_amp  size-sorted bucketed batching: crops sorted by padded volume,
                     chunked to <=B elements (and a padded-voxel cap), zero-padded
                     to the per-chunk max shape (multiple-of-`factor`), one forward
  seq_amp_compile    torch.compile(net, mode="reduce-overhead"); shapes quantised
                     to multiples of 32 to bound recompiles; compile warm-up timed
                     separately and EXCLUDED from the timed runs
  batch8_amp_compile combination (batch padded to a fixed B to stabilise shapes)

Two-phase protocol per patient: (A) build every element's input crop ONCE with
the exact production channel/prior code (timed; identical for all variants;
includes the npz cache IO + physics that the deployed loop also performs), then
(B) per-variant forwards + plan accumulation (timed). Per-patient total =
A + B, comparable to bench_deploy_time.py end-to-end numbers (crops make one
extra GPU->CPU->GPU round-trip here, ~negligible vs the forward).

Accuracy: for the first --acc-n elements of the first timed patient, each
variant's dose crop is compared to the seq_amp reference: max|diff| as % of the
reference crop max (flag if > 0.1%), plus the accumulated-plan max rel diff.
Padding regions are cropped away before comparison (verifying that batching +
padding does not leak into outputs).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n doserad --no-capture-output python -u scripts/bench_deploy_accel.py \
    [--task photon_ct,proton_ct] [--warmup 1] [--timed 2] [--acc-n 20] \
    [--out docs/reports/deploy_accel_results.md]
"""
from __future__ import annotations
import argparse, json, os, sys, time, statistics as stat
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
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
VOX_CAP = 4_000_000          # padded-voxel budget per batch (B*D*H*W), OOM-guarded anyway
COMPILE_FACTOR = 32          # shape quantisation for torch.compile variants


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _val_pids(cfg):
    return json.load(open(cfg["splits"]))[f"fold_{cfg['fold']}"]["val"]


# ============================================================== element builders
def setup_photon():
    from doserad.model.unet3d import DoseUNet3D
    from doserad.physics.machine import load_photon_machine
    cfg = yaml.safe_load(open(f"{CFG}/ftg_skinentry_photonct_f0.yaml"))
    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(DEV).eval()
    net.load_state_dict(torch.load(f"{WORKDIR}/runs/ftg_skinentry_photonct_f0/state.pt",
                                   map_location=DEV)["ema"])
    from doserad.data.dataset import DOSE_SCALE
    return dict(net=net, cfg=cfg, machine=load_photon_machine(MACHINE),
                dose_scale=float(DOSE_SCALE), factor=8, pids=_val_pids(cfg))


def build_photon(pid, S):
    from doserad.physics.channels import photon_channels
    from doserad.physics.density import hu_to_density
    from doserad.inference.pipeline import _normalize_gpu, _build_coords
    from doserad.beam.parse import load_photon_plan
    from doserad.io.mha import load_mha
    ct = load_mha(f"{PHOTON_ROOT}/{pid}/image/ct.mha")
    plan = load_photon_plan(Path(f"{PHOTON_ROOT}/{pid}/{pid}.json"))
    sync(); t0 = time.time()
    density = hu_to_density(ct.array, S["machine"].hu_anchors)
    coords = _build_coords(ct, "cuda" if torch.cuda.is_available() else "cpu")
    add_naive = bool(S["cfg"].get("add_naive", False))
    elems = []
    for beam in plan.beams:
        for cp in beam.control_points:
            crop, bbox = photon_channels(
                image=ct, machine=S["machine"], iso_xyz=beam.iso_center,
                gantry_deg=cp.gantry_angle,
                mlc_left=np.asarray(cp.mlc_left_int_mm),
                mlc_right=np.asarray(cp.mlc_right_int_mm),
                density_override=density, coords=coords,
                crop_margin=8, return_tensor=True)
            elems.append((_normalize_gpu(crop, add_naive).cpu(), bbox))
    del coords
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync()
    return elems, time.time() - t0, ct.array.shape


def setup_proton():
    from doserad.model.unet3d import DoseUNet3D
    from doserad.physics.proton_pb_gpu import ProtonMachineData
    from doserad.physics.machine import load_photon_machine
    from doserad.data.proton_dataset import PROTON_DOSE_SCALE
    cfg = yaml.safe_load(open(f"{CFG}/ft_skinentry_protonct_f0.yaml"))
    net = DoseUNet3D(in_ch=cfg["in_ch"], base=cfg["base_ch"], levels=cfg["levels"],
                     bottleneck=cfg.get("bottleneck", "plain")).to(DEV).eval()
    sd = torch.load(f"{WORKDIR}/runs/ft_skinentry_protonct_f0/state.pt", map_location=DEV)
    net.load_state_dict(sd.get("ema", sd.get("model")))
    return dict(net=net, cfg=cfg, machine=load_photon_machine(MACHINE),
                pm=ProtonMachineData(device=DEV), dose_scale=float(PROTON_DOSE_SCALE),
                factor=16, pids=_val_pids(cfg), cache=Path(cfg["cache_dir"]))


def build_proton(pid, S):
    import eval_protonmri_plan as epm
    from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry
    from doserad.physics.density import hu_to_density
    from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR
    from doserad.io.mha import load_mha
    ct = load_mha(f"{PROTON_ROOT}/{pid}/image/ct.mha")
    sync(); t0 = time.time()
    density = hu_to_density(ct.array, S["machine"].hu_anchors).astype(np.float32)
    plan = json.load(open(f"{PROTON_ROOT}/{pid}/{pid}.json"))
    rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
            (r["ray_source"], r["ray_target"], bl["energy"])
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}
    files = sorted(f for f in (S["cache"] / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
    elems = []
    for f in files:
        z = np.load(f); ch = z["channels"].astype(np.float32)
        bb = tuple(int(v) for v in z["bbox"])
        z0, z1, y0, y1, x0, x1 = bb
        sl = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
        b, r, l = (int(f.stem.split("_")[i][1:]) for i in range(3))
        src, tgt, e = rays[(b, r, l)]
        wepl_c = epm._wepl_on_density(ct, density, src, tgt, bb, S["pm"], DEV)
        pb = proton_pb_dose_gpu_skinentry(ct, src, tgt, e, out_bbox=bb, machine=S["pm"],
                                          density_override=density, device=DEV).astype(np.float32)
        inp = np.stack([density[sl], wepl_c, pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) \
            / _P_CH_SCALE_PRIOR[:, None, None, None]
        elems.append((torch.from_numpy(np.ascontiguousarray(inp.astype(np.float32))), bb))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync()
    return elems, time.time() - t0, ct.array.shape


# ============================================================== forward runners
def _pad_dims(shape, factor):
    d, h, w = shape
    return (-(-d // factor) * factor, -(-h // factor) * factor, -(-w // factor) * factor)


def run_seq(net, elems, plan_shape, *, amp, factor, dose_scale, acc_n=0, deadline=None):
    plan = np.zeros(plan_shape, np.float32); preds = {}
    mod = torch.zeros(1, dtype=torch.long, device=DEV)
    for i, (c, bb) in enumerate(elems):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"compile warm-up budget exceeded at elem {i}/{len(elems)}")
        d, h, w = c.shape[-3:]
        D, H, W = _pad_dims((d, h, w), factor)
        x = c.to(DEV, non_blocking=True)[None]
        if (D, H, W) != (d, h, w):
            x = F.pad(x, (0, W - w, 0, H - h, 0, D - d))
        with torch.no_grad(), torch.autocast("cuda", enabled=(amp and DEV != "cpu")):
            out = net(x, mod)
        arr = (out[0, 0, :d, :h, :w].float() / dose_scale).cpu().numpy()
        z0, z1, y0, y1, x0, x1 = bb
        plan[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += arr
        if i < acc_n:
            preds[i] = arr
    return plan, preds, {}


def _fwd_chunk(net, elems, chunk, plan, preds, *, amp, factor, dose_scale, acc_n,
               pad_batch_to=0, notes=None):
    dims = [_pad_dims(elems[i][0].shape[-3:], factor) for i in chunk]
    D = max(d[0] for d in dims); H = max(d[1] for d in dims); W = max(d[2] for d in dims)
    B = max(len(chunk), pad_batch_to)
    C = elems[chunk[0]][0].shape[0]
    try:
        xb = torch.zeros((B, C, D, H, W), dtype=torch.float32, device=DEV)
        for j, i in enumerate(chunk):
            c = elems[i][0]; d, h, w = c.shape[-3:]
            xb[j, :, :d, :h, :w] = c.to(DEV, non_blocking=True)
        mod = torch.zeros(B, dtype=torch.long, device=DEV)
        with torch.no_grad(), torch.autocast("cuda", enabled=(amp and DEV != "cpu")):
            yb = net(xb, mod)
        yb = (yb.float() / dose_scale).cpu().numpy()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
        if not (isinstance(ex, torch.cuda.OutOfMemoryError) or "out of memory" in str(ex).lower()):
            raise
        del xb
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            raise
        if notes is not None:
            notes.setdefault("oom_splits", 0); notes["oom_splits"] += 1
        mid = len(chunk) // 2
        _fwd_chunk(net, elems, chunk[:mid], plan, preds, amp=amp, factor=factor,
                   dose_scale=dose_scale, acc_n=acc_n, pad_batch_to=pad_batch_to, notes=notes)
        _fwd_chunk(net, elems, chunk[mid:], plan, preds, amp=amp, factor=factor,
                   dose_scale=dose_scale, acc_n=acc_n, pad_batch_to=pad_batch_to, notes=notes)
        return
    for j, i in enumerate(chunk):
        c, bb = elems[i]; d, h, w = c.shape[-3:]
        arr = yb[j, 0, :d, :h, :w]
        z0, z1, y0, y1, x0, x1 = bb
        plan[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += arr
        if i < acc_n:
            preds[i] = arr.copy()


def run_batched(net, elems, plan_shape, *, B, amp, factor, dose_scale, acc_n=0,
                pad_batch_to=0, deadline=None):
    order = sorted(range(len(elems)),
                   key=lambda i: int(np.prod(_pad_dims(elems[i][0].shape[-3:], factor))))
    chunks, cur, curmax = [], [], (0, 0, 0)
    for i in order:
        pd = _pad_dims(elems[i][0].shape[-3:], factor)
        nmx = tuple(max(a, b) for a, b in zip(curmax, pd))
        if cur and (len(cur) >= B or (len(cur) + 1) * int(np.prod(nmx)) > VOX_CAP):
            chunks.append(cur); cur, curmax = [i], pd
        else:
            cur.append(i); curmax = nmx
    if cur:
        chunks.append(cur)
    plan = np.zeros(plan_shape, np.float32); preds = {}; notes = {}
    for ci, ch in enumerate(chunks):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"compile warm-up budget exceeded at chunk {ci}/{len(chunks)}")
        _fwd_chunk(net, elems, ch, plan, preds, amp=amp, factor=factor,
                   dose_scale=dose_scale, acc_n=acc_n, pad_batch_to=pad_batch_to, notes=notes)
    notes["eff_batch"] = len(elems) / max(len(chunks), 1)
    return plan, preds, notes


# ============================================================== variant driver
VARIANTS = [
    ("seq_amp",            dict(mode="seq",   amp=True)),
    ("seq_fp32",           dict(mode="seq",   amp=False)),
    ("batch4_amp",         dict(mode="batch", B=4,  amp=True)),
    ("batch8_amp",         dict(mode="batch", B=8,  amp=True)),
    ("batch16_amp",        dict(mode="batch", B=16, amp=True)),
    ("seq_amp_compile",    dict(mode="seq",   amp=True, compile=True)),
    ("batch8_amp_compile", dict(mode="batch", B=8,  amp=True, compile=True)),
]


def bench_task(task, n_warmup, n_timed, acc_n, only=None):
    print(f"\n================ {task} ================", flush=True)
    variants = VARIANTS if not only else [(n, v) for n, v in VARIANTS if n in only]
    S = setup_photon() if task == "photon_ct" else setup_proton()
    build = build_photon if task == "photon_ct" else build_proton
    pids = S["pids"][: n_warmup + n_timed]
    patients = []
    for pid in pids:
        elems, build_s, shape = build(pid, S)
        print(f"  [build] {pid}: {len(elems)} elems in {build_s:.2f}s", flush=True)
        patients.append(dict(pid=pid, elems=elems, build_s=build_s, shape=shape))
    warm, timed = patients[:n_warmup], patients[n_warmup:]

    compiled_net = None
    ref_preds, ref_plan = None, None
    rows = []
    for vname, v in variants:
        factor = COMPILE_FACTOR if v.get("compile") else S["factor"]
        net = S["net"]
        row = dict(variant=vname, notes=[])
        if v.get("compile"):
            if compiled_net is None:
                try:
                    torch._dynamo.config.cache_size_limit = 128
                    compiled_net = torch.compile(S["net"], mode="reduce-overhead")
                except Exception as ex:  # noqa: BLE001
                    row["error"] = f"torch.compile failed: {ex}"
                    rows.append(row); print(f"  [{vname}] {row['error']}", flush=True)
                    compiled_net = "FAILED"; continue
            if compiled_net == "FAILED":
                row["error"] = "torch.compile failed earlier"; rows.append(row); continue
            net = compiled_net

        def _run(p, collect=0, deadline=None):
            kw = dict(amp=v["amp"], factor=factor, dose_scale=S["dose_scale"], acc_n=collect,
                      deadline=deadline)
            if v["mode"] == "seq":
                return run_seq(net, p["elems"], p["shape"], **kw)
            return run_batched(net, p["elems"], p["shape"], B=v["B"],
                               pad_batch_to=(v["B"] if v.get("compile") else 0), **kw)

        try:
            # warm-up patient (compile happens here; timed separately, budget-capped)
            sync(); tw = time.time()
            dl = (tw + 1200) if v.get("compile") else None
            for p in warm:
                _run(p, deadline=dl)
            sync(); warm_s = time.time() - tw
            if v.get("compile"):
                row["compile_warmup_s"] = warm_s
                print(f"  [{vname}] compile/warm-up pass: {warm_s:.1f}s", flush=True)
            ts, notes_all = [], {}
            preds_first, plan_first = None, None
            for k, p in enumerate(timed):
                collect = acc_n if k == 0 else 0
                sync(); t0 = time.time()
                plan, preds, notes = _run(p, collect=collect)
                sync(); dt = time.time() - t0
                ts.append((dt, len(p["elems"])))
                if k == 0:
                    preds_first, plan_first = preds, plan
                for kk, vv in notes.items():
                    notes_all[kk] = vv
                print(f"  [{vname}] {p['pid']}: fwd {dt:6.2f}s  {len(p['elems'])} elems"
                      f"  {dt / len(p['elems']) * 1000:7.2f} ms/elem", flush=True)
            row["per_elem_ms"] = stat.mean(dt / n * 1000 for dt, n in ts)
            row["fwd_s"] = stat.mean(dt for dt, _ in ts)
            row["total_s"] = stat.mean(dt + p["build_s"] for (dt, _), p in zip(ts, timed))
            if "eff_batch" in notes_all:
                row["notes"].append(f"eff.batch {notes_all['eff_batch']:.1f}")
            if notes_all.get("oom_splits"):
                row["notes"].append(f"OOM-splits {notes_all['oom_splits']}")
            # ---- accuracy vs seq_amp reference ----
            if vname == "seq_amp":
                ref_preds, ref_plan = preds_first, plan_first
            elif ref_preds:
                rel = [float(np.abs(preds_first[i] - ref_preds[i]).max()
                             / max(float(ref_preds[i].max()), 1e-12) * 100)
                       for i in ref_preds if i in preds_first]
                row["max_rel_diff_pct"] = max(rel) if rel else float("nan")
                row["plan_rel_diff_pct"] = float(np.abs(plan_first - ref_plan).max()
                                                 / max(float(ref_plan.max()), 1e-12) * 100)
        except Exception as ex:  # noqa: BLE001
            import traceback; traceback.print_exc()
            row["error"] = str(ex)
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    build_s = stat.mean(p["build_s"] for p in timed)
    n_elems = stat.mean(len(p["elems"]) for p in timed)
    # free the compiled net + its CUDA-graph private pools before the next task
    # (torch.compile reduce-overhead leaks ~19GB of graph pools across tasks otherwise)
    del compiled_net, patients, warm, timed
    if torch.cuda.is_available():
        import gc
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return rows, build_s, n_elems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="photon_ct,proton_ct")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--acc-n", type=int, default=20)
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of variant names to run (default: all)")
    ap.add_argument("--out", default=f"{os.path.dirname(_HERE)}/docs/reports/deploy_accel_results.md")
    ap.add_argument("--gpu-frac", type=float, default=0.75,
                    help="cap this process's GPU memory fraction so a too-large batch raises a "
                         "catchable OOM instead of a hard SIGKILL when sharing VRAM with a co-tenant")
    a = ap.parse_args()

    if torch.cuda.is_available() and a.gpu_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac, 0)

    import subprocess
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"GPU(s):\n{gpu}\nCUDA_VISIBLE_DEVICES={vis} device={DEV} torch={torch.__version__}", flush=True)

    Path(os.path.dirname(a.out)).mkdir(parents=True, exist_ok=True)

    def write_report(sections):
        with open(a.out, "w") as fh:
            fh.write("# DoseRAD2026 — Deployment Acceleration Prototypes (CT tasks)\n\n")
            fh.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}\n")
            fh.write(f"- Hardware: {gpu.splitlines()[0] if gpu else 'n/a'} (CUDA_VISIBLE_DEVICES={vis})\n")
            fh.write(f"- torch {torch.__version__}; per-process GPU fraction cap {a.gpu_frac}\n")
            fh.write(f"- Timed: {a.timed} fold-0 val patients (+{a.warmup} warm-up, excluded); "
                     f"accuracy on first {a.acc_n} elements of the first timed patient vs seq_amp.\n")
            fh.write("- NOTE: the deployed 'FP32 baseline' forwards already run under torch.autocast "
                     "(photon-CT additionally streams batches of <=8). seq_amp = batch-1 autocast; "
                     "seq_fp32 = autocast disabled (cuDNN TF32-conv defaults).\n")
            for s in sections:
                fh.write(s + "\n")

    only = set(x.strip() for x in a.variants.split(",")) if a.variants else None
    sections = []
    for tk in [t.strip() for t in a.task.split(",")]:
        rows, build_s, n_elems = bench_task(tk, a.warmup, a.timed, a.acc_n, only=only)
        base = next((r for r in rows if r["variant"] == "seq_amp" and "fwd_s" in r), None)
        lines = [f"\n## {tk}\n",
                 f"- avg elements/patient: {n_elems:.0f}; channel/prior build (identical for all "
                 f"variants, incl. cache IO + physics): **{build_s:.2f} s/patient**",
                 "",
                 "| variant | per-elem fwd (ms) | fwd (s/patient) | total = build+fwd (s/patient) "
                 "| speedup vs seq_amp (fwd) | speedup (total) | max elem rel-diff (% of crop max) "
                 "| plan rel-diff (%) | notes |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['variant']} | - | - | - | - | - | - | - | ERROR: {r['error'][:120]} |")
                continue
            sp_f = base["fwd_s"] / r["fwd_s"] if base else float("nan")
            sp_t = base["total_s"] / r["total_s"] if base else float("nan")
            acc = ("ref" if r["variant"] == "seq_amp"
                   else f"{r.get('max_rel_diff_pct', float('nan')):.4f}"
                        + (" **FLAG**" if r.get("max_rel_diff_pct", 0) > 0.1 else ""))
            pacc = "ref" if r["variant"] == "seq_amp" else f"{r.get('plan_rel_diff_pct', float('nan')):.4f}"
            notes = "; ".join(r["notes"])
            if "compile_warmup_s" in r:
                notes = (notes + "; " if notes else "") + f"compile warm-up {r['compile_warmup_s']:.0f}s (excluded)"
            lines.append(f"| {r['variant']} | {r['per_elem_ms']:.2f} | {r['fwd_s']:.2f} | "
                         f"{r['total_s']:.2f} | {sp_f:.2f}x | {sp_t:.2f}x | {acc} | {pacc} | {notes} |")
        sections.append("\n".join(lines))
        print(sections[-1], flush=True)
        write_report(sections)          # incremental: survive a later-task crash
        print(f"[written-partial] {a.out} (through {tk})", flush=True)

    write_report(sections)
    print(f"\n[written] {a.out}", flush=True)


if __name__ == "__main__":
    main()
