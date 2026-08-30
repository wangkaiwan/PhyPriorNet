"""Precompute the Tier-2 AAA prior (aaa_prior_dose) for every cached photon CP and
store it in a PARALLEL cache, aligned to the existing channels-cache bbox so it drops
in as a 6th input channel (v12). The AAA prior needs full CT + geometry, so (unlike
naive) it cannot be derived from the 5 cached channels on the fly → precompute here.

For each patient with a channels cache, for each `<b>_<cp>.npz`, read its bbox and
compute aaa_prior_dose(out_bbox=bbox); save {"aaa", "bbox"} to AAA_CACHE/<pid>/<b>_<cp>.npz.
Skips files already done (resumable). Shardable for parallelism: --shard i --nshards N
splits PATIENTS across processes.

  python scripts/build_aaa_cache.py [--device cuda] [--shard 0 --nshards 4] [--max-patients N]
"""
import argparse
import time
from pathlib import Path

import numpy as np

from doserad.beam.parse import load_photon_plan
from doserad.io.mha import load_mha
from doserad.physics.density import hu_to_density
from doserad.physics.geometry import beam_source_pos, beam_basis
from doserad.physics.machine import load_photon_machine
from doserad.physics.priors.pencil_beam_aaa import load_aaa_kernel, aaa_prior_dose

ROOT = "/data/kwang/DoseRad2026_raw/photon/training"
CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon")
AAA_CACHE = Path("/home/kaiwang/doserad2026_workdir/cache/crops/photon_aaa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--out", default=str(AAA_CACHE))
    args = ap.parse_args()
    out_root = Path(args.out)

    machine = load_photon_machine(f"{ROOT}/beam_parameters.json")
    kernel = load_aaa_kernel()
    pids = sorted(d.name for d in CACHE.iterdir() if d.is_dir())
    pids = pids[args.shard::args.nshards]
    if args.max_patients:
        pids = pids[:args.max_patients]
    print(f"[shard {args.shard}/{args.nshards}] {len(pids)} patients", flush=True)

    n_done = n_skip = 0
    t0 = time.time()
    for pid in pids:
        cfiles = sorted((CACHE / pid).glob("*.npz"))
        if not cfiles:
            continue
        plan = load_photon_plan(f"{ROOT}/{pid}/{pid}.json")
        ct = load_mha(f"{ROOT}/{pid}/image/ct.mha")
        density = hu_to_density(ct.array, machine.hu_anchors).astype(np.float32)
        cp_by = {(b.beam_idx, cp.cp_idx): (b, cp)
                 for b in plan.beams for cp in b.control_points}
        (out_root / pid).mkdir(parents=True, exist_ok=True)
        for f in cfiles:
            outf = out_root / pid / f.name
            if outf.exists():
                n_skip += 1
                continue
            bidx, cidx = (int(x) for x in f.stem.split("_"))
            beam, cp = cp_by[(bidx, cidx)]
            z = np.load(f)
            bbox = tuple(int(v) for v in z["bbox"])
            iso = np.asarray(beam.iso_center, dtype=np.float64)
            src = beam_source_pos(iso, machine.sad_mm, cp.gantry_angle)
            ax, uh, vh = beam_basis(cp.gantry_angle)
            aaa = aaa_prior_dose(density, ct.spacing, ct.origin, src, ax, uh, vh, iso,
                                 machine, np.asarray(cp.mlc_left_int_mm),
                                 np.asarray(cp.mlc_right_int_mm), kernel,
                                 out_bbox=bbox, device=args.device)
            np.savez_compressed(outf, aaa=aaa.astype(np.float32),
                                bbox=np.asarray(bbox, np.int32))
            n_done += 1
        dt = time.time() - t0
        print(f"  {pid}: done={n_done} skip={n_skip} ({dt/60:.1f} min)", flush=True)
    print(f"[shard {args.shard}] FINISHED done={n_done} skip={n_skip} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
