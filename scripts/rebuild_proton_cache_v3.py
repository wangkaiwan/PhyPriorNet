"""Rebuild the proton training WEPL/prior caches under the ENGINE-V2 (parallel-column BEV) physics.

Engine v2's plug-in gamma sits −2.3 below deployed because the net was trained on from-source-march
channels. Fix = train/deploy consistency at the V3 convention: for every cached beamlet crop
(proton_ssd bboxes, GT dose untouched), recompute ch1 WEPL + ch2 PB with wepl_beam_ortho and write
two parallel dirs consumed via the trainer's wepl_dir/prior_dir hooks:
    proton_weplfix_v3/<pid>/B*_R*_L*.npz  {wepl}      (float16, source-ref V3)
    proton_prior_v3/<pid>/B*_R*_L*.npz    {pb_prior}  (float16, Gy, V3 w_skin/entered)

Then finetune all75_r2_ft on (cache_dir unchanged, wepl_dir/prior_dir -> v3) — the engine-v2 net.
DOSERAD_SHARD=k/N; usage: CUDA_VISIBLE_DEVICES=1 python scripts/rebuild_proton_cache_v3.py [pids...]
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.machine import load_photon_machine
from doserad.physics.proton_pb_gpu import ProtonMachineData, _interp_1d, _compute_ssd
from accel.proton_build_ray import _SSD_DENSITY_THRESHOLD
import importlib.util
_sp = importlib.util.spec_from_file_location("bw", str(Path(__file__).resolve().parents[1] / "accel" / "exp_batched_wepl.py"))
bw = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bw)

ROOT = "/data/kwang/DoseRad2026_raw/proton/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
OUT_W = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_weplfix_v3")
OUT_P = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_prior_v3")
DEV = "cuda"
FORCE = bool(os.environ.get("DOSERAD_FORCE"))


@torch.no_grad()
def process(pid, machine, pm):
    pdir = Path(ROOT) / pid
    ct = load_mha(pdir / "image" / "ct.mha")
    dens = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
    dens_t = torch.as_tensor(dens, device=DEV)
    dens5 = dens_t.view(1, 1, *dens.shape)
    plan = json.load(open(pdir / f"{pid}.json"))
    (OUT_W / pid).mkdir(parents=True, exist_ok=True)
    (OUT_P / pid).mkdir(parents=True, exist_ok=True)
    files = sorted(f.name for f in (CACHE / pid).glob("B*_R*_L*.npz") if ".tmp" not in f.name)
    by_beam = {}
    for fn in files:
        b, r, l = (int(x[1:]) for x in fn[:-4].split("_"))
        by_beam.setdefault(b, []).append((fn, r, l))
    n = 0
    for b, lst in by_beam.items():
        beam = plan["beams"][b]
        bd = np.asarray(beam["rays"][0]["ray_target"], np.float64) - np.asarray(beam["rays"][0]["ray_source"], np.float64)
        todo = [x for x in lst if FORCE or not ((OUT_W / pid / x[0]).exists() and (OUT_P / pid / x[0]).exists())]
        if not todo:
            continue
        metas, coords = [], []
        for fn, r, l in todo:
            z = np.load(CACHE / pid / fn)
            bb = tuple(int(v) for v in z["bbox"])
            metas.append((fn, r, l, bb, float(z["energy"])))
            coords.append(bw.coords_of_bbox(bb, ct.spacing, ct.origin, DEV))
        weps = bw.wepl_beam_ortho(dens5, ct.spacing, ct.origin, bd, coords)
        ssd_cache = {}
        for (fn, r, l, bb, energy), c, (w_src, w_skin, entered) in zip(metas, coords, weps):
            ray = beam["rays"][r]
            if r not in ssd_cache:
                axis_f = np.asarray(ray["ray_target"], np.float64) - np.asarray(ray["ray_source"], np.float64)
                axis_f = axis_f / np.linalg.norm(axis_f)
                ssd_cache[r] = (axis_f, _compute_ssd(dens_t, ct.spacing, ct.origin, ray["ray_source"],
                                                     axis_f, pm.sad, DEV, threshold=_SSD_DENSITY_THRESHOLD))
            axis_f, ssd = ssd_cache[r]
            src_t = torch.as_tensor(ray["ray_source"], device=DEV, dtype=torch.float32)
            axis_t = torch.as_tensor(axis_f, device=DEV, dtype=torch.float32)
            rel = c - src_t
            along = (rel * axis_t).sum(-1, keepdim=True) * axis_t
            lat = torch.linalg.norm(rel - along, dim=-1)
            rad = w_skin * 10.0
            eidx = pm.energy_index(energy); nn_ = int(pm.lengths[eidx].item())
            depths = pm.depths[eidx, :nn_]; idd = pm.conversion_factor * pm.idd[eidx, :nn_]
            sigma_d = pm.sigma[eidx, :nn_]; offset = float(pm.offset[eidx].item())
            sig_ini_sq = pm.sigma_ini(eidx, ssd) ** 2
            eff = depths + offset
            d = rad.reshape(-1)
            idd_v = _interp_1d(d, eff, idd); sig_v = _interp_1d(d, eff, sigma_d)
            oor = (d > eff[-1]) | (d < eff[0])
            idd_v = torch.where(oor, torch.zeros_like(idd_v), idd_v)
            sigma_sq = sig_v ** 2 + sig_ini_sq
            lateral = torch.exp(-(lat.reshape(-1) ** 2) / (2.0 * sigma_sq)) / (2.0 * torch.pi * sigma_sq)
            pb = (lateral * idd_v).reshape(rad.shape).clamp_min(0.0) * entered
            for out_dir, key, arr in ((OUT_W, "wepl", w_src), (OUT_P, "pb_prior", pb)):
                tmp = (out_dir / pid / fn).with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, **{key: arr.cpu().numpy().astype(np.float16)})
                os.replace(tmp, out_dir / pid / fn)
            n += 1
        del weps, coords
        torch.cuda.empty_cache()
    return n


def main(pids):
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    pm = ProtonMachineData(device=DEV)
    if not pids:
        pids = sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    shard = os.environ.get("DOSERAD_SHARD")
    if shard:
        k, N = (int(x) for x in shard.split("/"))
        pids = [p for i, p in enumerate(pids) if i % N == k]
    print(f"[v3-cache] {len(pids)} pids -> {OUT_W} + {OUT_P}", flush=True)
    for pid in pids:
        t0 = time.time()
        n = process(pid, machine, pm)
        print(f"  {pid}: {n} beamlets in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
