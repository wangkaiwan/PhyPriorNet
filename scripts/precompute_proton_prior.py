"""Precompute the pyRadPlan PB-prior cache for proton beamlets (the WITH-prior arm of the
v13-style prior A/B). SEPARATE cache from the no-prior channels cache (precompute_proton.py),
because pyRadPlan needs the `pyradplan` conda env while the channels need `doserad` — mirrors the
photon aaa_cache_dir split. The dataset stacks pb_prior as an extra channel.

For each beamlet it reads the bbox from the EXISTING no-prior npz (so the crop is IDENTICAL),
computes the pyRadPlan Hong PB dose (verified corr 0.966 vs MC GT), crops, saves pb_prior fp16.

RUNS IN `pyradplan` ENV. Shard across CPU cores by launching many processes:
    for i in $(seq 0 23); do conda run -n pyradplan python scripts/precompute_proton_prior.py --shard $i/24 & done
~11 s/beamlet single-core -> ~1080*75/24 cores ~ 10 h. CPU-only (no GPU).
"""
from __future__ import annotations

import os
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Reference pencil-beam implementation used to validate our GPU port (r=0.995).
# Point PYRADPLAN_BASELINE at your checkout if you want to run the cross-check;
# the cache build itself uses our self-contained GPU engine and does not need it.
_baseline = os.environ.get("PYRADPLAN_BASELINE")
if _baseline:
    sys.path.insert(0, _baseline)
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import baseline_protons as B
from pyRadPlan.machines import load_from_name

ROOT = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/proton/training")
BEAM_PARAMS = (os.environ.get("DATA_ROOT", "/data/DoseRad2026_raw") + "/beam_parameters.json")
NOPRIOR_CACHE = (os.environ.get("WORKDIR", "./workdir") + "/cache/crops/proton")
PRIOR_CACHE = (os.environ.get("WORKDIR", "./workdir") + "/cache/crops/proton_prior")


def _patient_setup(pid):
    base = Path(ROOT) / pid
    ct = B.ct_from_file(str(base / B.DIR_IMAGE / B.CT_NAME))
    pln_json = json.load(open(base / f"{pid}.json"))
    bp = json.load(open(BEAM_PARAMS))
    hlut = np.array([tuple(e.values()) for e in bp["hu_to_density"]["entries"]], dtype=float)
    machine = load_from_name("protons", "Generic")
    energies = np.array(sorted(machine.energies), dtype=np.float64)
    sad = machine.sad
    cst = B.StructureSet(vois=[], ct_image=ct); cst.create_body_seg()
    pln = B.IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_dose_calc = {"dose_grid": {"resolution": ct.resolution}, "air_offset_correction": True,
                          "geometric_lateral_cutoff": 25.0, "trace_on_dose_grid": True, "hlut": hlut}
    return ct, pln_json, cst, pln, sad, energies


def process_patient(pid):
    npdir = Path(NOPRIOR_CACHE) / pid
    if not npdir.exists():
        return 0, 0
    odir = Path(PRIOR_CACHE) / pid; odir.mkdir(parents=True, exist_ok=True)
    ct, pln_json, cst, pln, sad, energies = _patient_setup(pid)
    beams = {b["beam_idx"]: b for b in pln_json["beams"]}
    done = skip = 0
    for npz in sorted(npdir.glob("B*_R*_L*.npz")):
        out = odir / npz.name
        if out.exists():
            skip += 1; continue
        # parse B{b}_R{r}_L{l}
        stem = npz.stem  # B0_R0_L0
        b = int(stem.split("_")[0][1:]); r = int(stem.split("_")[1][1:]); l = int(stem.split("_")[2][1:])
        bbox = np.load(npz)["bbox"]
        z0, z1, y0, y1, x0, x1 = (int(v) for v in bbox)
        beam = beams[b]
        ray = next(rr for rr in beam["rays"] if rr["ray_idx"] == r)
        bl = next(bb for bb in ray["beamlets"] if bb["beamlet_idx"] == l)
        pb_dose, _ = B._compute_beamlet_dose(ct, cst, pln, beam, ray, bl, sad, energies)
        pb = sitk.GetArrayFromImage(pb_dose).astype(np.float32)[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, pb_prior=pb.astype(np.float16))
        tmp.replace(out)
        done += 1
    return done, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--pids", default=None)
    a = ap.parse_args()
    pids = a.pids.split(",") if a.pids else sorted(p.name for p in Path(ROOT).iterdir() if p.is_dir())
    i, N = (int(x) for x in a.shard.split("/"))
    pids = pids[i::N]
    print(f"[prior shard {i}/{N}] {len(pids)} patients: {pids[:3]}...", flush=True)
    for k, pid in enumerate(pids):
        t = time.time()
        try:
            d, s = process_patient(pid)
            print(f"  [{k+1}/{len(pids)}] {pid}: +{d} (skip {s}) [{time.time()-t:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k+1}/{len(pids)}] {pid}: ERROR {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
