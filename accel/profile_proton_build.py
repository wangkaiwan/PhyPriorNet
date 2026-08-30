"""Acceleration sprint — Phase A step 0: component-level profile of the PROTON deploy build.

Per beamlet the deployed path does: load crop npz -> WEPL ray-march (_wepl_on_density) ->
GPU PB prior (proton_pb_dose_gpu_skinentry) -> stack/normalize -> net forward. This script
times each component over N beamlets of one patient (GPU-synced timers), and additionally
quantifies the per-RAY dedup opportunity: how much do ray-sibling beamlets (same B,R; L0/L1)
share (bbox overlap, WEPL-on-union feasibility)?

Run (GPU1): CUDA_VISIBLE_DEVICES=1 conda run -n doserad python -u accel/profile_proton_build.py
"""
from __future__ import annotations

import sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.eval_protonmri_plan as epm  # production WEPL helper  # noqa: E402
from doserad.io.mha import load_mha  # noqa: E402
from doserad.data.proton_dataset import PROTON_DOSE_SCALE, _P_CH_SCALE_PRIOR  # noqa: E402
from doserad.model.unet3d import DoseUNet3D  # noqa: E402
from doserad.physics.proton_pb_gpu import ProtonMachineData  # noqa: E402
from doserad.physics.proton_pb_gpu_skinentry import proton_pb_dose_gpu_skinentry  # noqa: E402
from doserad.physics.density import hu_to_density  # noqa: E402
from doserad.physics.machine import load_photon_machine  # noqa: E402
import json  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd")
ROOT = Path("/data/kwang/DoseRad2026_raw/proton/training")
PID = "1ABB006"
N = 60  # beamlets to time (after 5 warmup)


def sync():
    torch.cuda.synchronize()


def main():
    pm = ProtonMachineData(device=DEV)
    machine = load_photon_machine("/data/kwang/DoseRad2026_raw/beam_parameters.json")
    ct = load_mha(ROOT / PID / "image" / "ct.mha")
    density = hu_to_density(ct.array.astype(np.float32), machine.hu_anchors)
    plan = json.load(open(ROOT / PID / f"{PID}.json"))
    net = DoseUNet3D(in_ch=5, base=48, levels=4, bottleneck="dilated").to(DEV).eval()
    st = torch.load("/home/kaiwang/doserad2026_workdir/runs/ft_skinentry_protonct_f0/state.pt",
                    map_location=DEV)
    net.load_state_dict(st.get("ema", st.get("model")))
    mod = torch.zeros(1, dtype=torch.long, device=DEV)

    # ray table from the plan json
    rays = {(b["beam_idx"], r["ray_idx"], bl["beamlet_idx"]):
            dict(src=np.array(r["ray_source"], np.float64),
                 tgt=np.array(r["ray_target"], np.float64), e=float(bl["energy"]))
            for b in plan["beams"] for r in b["rays"] for bl in r["beamlets"]}

    files = sorted(f for f in (CACHE / PID).glob("B*_R*_L*.npz") if ".tmp" not in f.name)[: N + 5]
    T = defaultdict(float); n = 0
    bb_by_ray = defaultdict(list)
    for i, f in enumerate(files):
        b, r, l = (int(f.stem.split("_")[k][1:]) for k in range(3))
        ray = rays[(b, r, l)]
        t0 = time.perf_counter()
        z = np.load(f); ch = z["channels"].astype(np.float32)
        bb = tuple(int(v) for v in z["bbox"])
        t1 = time.perf_counter()
        wepl_c = epm._wepl_on_density(ct, density, ray["src"], ray["tgt"], bb, pm, DEV); sync()
        t2 = time.perf_counter()
        pb = proton_pb_dose_gpu_skinentry(ct, ray["src"], ray["tgt"], ray["e"], out_bbox=bb,
                                          machine=pm, density_override=density, device=DEV); sync()
        t3 = time.perf_counter()
        z0, z1, y0, y1, x0, x1 = bb
        dens_c = density[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        inp = np.stack([dens_c, wepl_c, pb * PROTON_DOSE_SCALE, ch[2], ch[3]], 0) \
            / _P_CH_SCALE_PRIOR[:, None, None, None]
        x = torch.from_numpy(inp.astype(np.float32)[None]).to(DEV)
        Z, Y, X = x.shape[-3:]
        x = F.pad(x, (0, (-X) % 16, 0, (-Y) % 16, 0, (-Z) % 16))
        t4 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda"):
            _ = net(x, mod)
        sync()
        t5 = time.perf_counter()
        if i >= 5:  # skip warmup
            T["io_npz"] += t1 - t0; T["wepl"] += t2 - t1; T["pb_prior"] += t3 - t2
            T["stack_h2d"] += t4 - t3; T["forward"] += t5 - t4; n += 1
        bb_by_ray[(b, r)].append(bb)

    tot = sum(T.values())
    print(f"\n=== PROTON deploy build profile ({PID}, {n} beamlets, ms/beamlet) ===")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s}: {v / n * 1e3:7.1f} ms  ({v / tot * 100:4.1f}%)")
    print(f"  TOTAL     : {tot / n * 1e3:7.1f} ms/beamlet")

    # per-ray dedup: bbox overlap between ray siblings
    ious = []; same = 0
    for (b, r), bbs in bb_by_ray.items():
        if len(bbs) < 2:
            continue
        a, c = bbs[0], bbs[1]
        if a == c:
            same += 1
        inter = max(0, min(a[1], c[1]) - max(a[0], c[0])) * max(0, min(a[3], c[3]) - max(a[2], c[2])) \
            * max(0, min(a[5], c[5]) - max(a[4], c[4]))
        va = (a[1] - a[0]) * (a[3] - a[2]) * (a[5] - a[4]); vc = (c[1] - c[0]) * (c[3] - c[2]) * (c[5] - c[4])
        ious.append(inter / max(va + vc - inter, 1))
    if ious:
        print(f"\n=== per-RAY sibling stats ({len(ious)} rays with 2 beamlets) ===")
        print(f"  identical bbox: {same}/{len(ious)};  bbox IoU mean {np.mean(ious):.2f} min {np.min(ious):.2f}")
        print("  -> WEPL on the UNION bbox once per ray, slice per beamlet =",
              "TRIVIAL (bboxes ~identical)" if np.mean(ious) > 0.8 else "needs union-crop handling")


if __name__ == "__main__":
    main()
