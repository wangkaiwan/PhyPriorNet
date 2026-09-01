"""Proton pencil-beam (pyRadPlan) prior generator for DoseRAD2026 Phase-2.

Wraps the challenge org's `pyradplan-pb-baseline/baseline_protons.py` into a reusable CLI that
computes the analytical Hong pencil-beam dose for one beamlet (or all beamlets of a ray/beam),
optionally writes the `.mha` and compares to the Monte-Carlo GT (corr + Bragg-peak offset).

This is BOTH (a) the standalone proton baseline and (b) the GT-aligned PRIOR we will cache for
residual DL (mirrors the photon `naive_dose` recipe). VERIFIED 2026-06-19 on 1ABB006 B0R0L0:
corr 0.966 vs GT, Bragg peak within 1 voxel, ~11 s/beamlet on CPU.

RUNS IN THE `pyradplan` CONDA ENV (pyRadPlan 0.3.5), NOT `doserad`:
    conda run -n pyradplan python scripts/proton_pb_beamlet.py --patient 1ABB006 --beam 0 --ray 0 --beamlet 0 --compare

DATA-FORMAT NOTE: our proton plan JSON has NO "SAD" key (baseline expects one) → we use the
pyRadPlan Generic-machine SAD (10000 mm), which the baseline would override to anyway.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

BASELINE_REPO = "/home/kaiwang/project/pyradplan-pb-baseline"
ROOT = "/data/kwang/DoseRad2026_raw"
sys.path.insert(0, BASELINE_REPO)
import baseline_protons as B                       # noqa: E402
from pyRadPlan.machines import load_from_name      # noqa: E402


def _setup(pid, modality, split):
    base = Path(ROOT) / modality / split / pid
    ct = B.ct_from_file(str(base / B.DIR_IMAGE / B.CT_NAME))
    pln_json = json.load(open(base / f"{pid}.json"))
    beam_params = json.load(open(Path(ROOT) / B.BEAM_PARAMS_FILENAME))
    hlut = np.array([tuple(e.values()) for e in beam_params["hu_to_density"]["entries"]], dtype=float)
    machine = load_from_name("protons", "Generic")
    energies = np.array(sorted(machine.energies), dtype=np.float64)
    sad = machine.sad   # JSON has no SAD; baseline overrides to machine.sad anyway
    cst = B.StructureSet(vois=[], ct_image=ct); cst.create_body_seg()
    pln = B.IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_dose_calc = {"dose_grid": {"resolution": ct.resolution}, "air_offset_correction": True,
                          "geometric_lateral_cutoff": 25.0, "trace_on_dose_grid": True, "hlut": hlut}
    return base, ct, pln_json, cst, pln, sad, energies


def _one(ct, cst, pln, beam, ray, bl, sad, energies):
    pb_dose, energy = B._compute_beamlet_dose(ct, cst, pln, beam, ray, bl, sad, energies)
    return pb_dose, energy


def _compare(pb_dose, gt_path):
    pb = sitk.GetArrayFromImage(pb_dose).astype(np.float32)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.float32)
    m = gt > 0.01 * gt.max()
    corr = float(np.corrcoef(pb[m], gt[m])[0, 1]) if m.sum() > 10 else float("nan")
    off = np.subtract(np.unravel_index(pb.argmax(), pb.shape), np.unravel_index(gt.argmax(), gt.shape))
    return corr, off, float(pb.max()), float(gt.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--modality", default="proton")
    ap.add_argument("--split", default="training")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--ray", type=int, default=0)
    ap.add_argument("--beamlet", type=int, default=0, help="-1 = all beamlets of the ray")
    ap.add_argument("--out-dir", default=None, help="write PB dose .mha here (skip if unset)")
    ap.add_argument("--compare", action="store_true", help="report corr + Bragg offset vs GT")
    a = ap.parse_args()

    base, ct, pln_json, cst, pln, sad, energies = _setup(a.patient, a.modality, a.split)
    beam = pln_json["beams"][a.beam]; ray = beam["rays"][a.ray]
    bls = ray["beamlets"] if a.beamlet < 0 else [ray["beamlets"][a.beamlet]]
    print(f"{a.patient} B{beam['beam_idx']} R{ray['ray_idx']} gantry={beam['gantry_angle']} "
          f"| {len(bls)} beamlet(s) | machine SAD={sad}", flush=True)

    for bl in bls:
        t = time.time()
        pb_dose, energy = _one(ct, cst, pln, beam, ray, bl, sad, energies)
        b_idx, r_idx, l_idx = beam["beam_idx"], ray["ray_idx"], bl["beamlet_idx"]
        msg = f"  B{b_idx}R{r_idx}L{l_idx} E={energy:.2f}MeV (json {bl['energy']:.2f}) [{time.time()-t:.1f}s]"
        if a.out_dir:
            od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
            fn = od / f"{a.patient}_{a.modality}_{a.split}_PB_Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
            sitk.WriteImage(pb_dose, str(fn), useCompression=True); msg += f" -> {fn.name}"
        if a.compare:
            gt = base / B.DIR_DOSE / f"Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
            if gt.exists():
                corr, off, pbmx, gtmx = _compare(pb_dose, gt)
                msg += f" | corr {corr:.3f} bragg_off {tuple(int(v) for v in off)} max PB {pbmx:.3g}/GT {gtmx:.3g}"
            else:
                msg += " | (no GT file)"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
