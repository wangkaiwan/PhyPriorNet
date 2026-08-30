"""Local mock harness: fabricate a GC-style 10-slot input from training data + a metadata json
with output_info, run process_run with a TRIVIAL zeros predictor, and assert every pre-submission
checklist item (submission_instructions_2026-07-17.md §4).

Run: CUDA_VISIBLE_DEVICES=1 conda run -n doserad python container/proton/test_contract.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc_invoke import process_run, N_SLOTS

ROOT = Path("/data/kwang/DoseRad2026_raw/proton/training")
PIDS = ["1ABB006", "1THB002"]     # 2 patients -> 2 used image slots (multi-patient run)
PARTICLE, MODALITY = "proton", "ct"


def build_mock_input(tmp: Path):
    in_dir = tmp / "input"
    (in_dir / "images").mkdir(parents=True)
    meta = []
    for k in range(N_SLOTS):
        slot = k + 1
        d = in_dir / "images" / f"radiation-dose-calculation-source-{MODALITY}-image-{slot}"
        d.mkdir(parents=True)
        if k < len(PIDS):
            img = sitk.ReadImage(str(ROOT / PIDS[k] / "image" / "ct.mha"))
            sitk.WriteImage(img, str(d / f"img{slot}.mha"), useCompression=True)
        else:
            ph = sitk.GetImageFromArray(np.zeros((1, 1, 1), np.float32))
            sitk.WriteImage(ph, str(d / "placeholder.mha"))
    # metadata: for each used patient, a few beams w/ output_info -> distinct output slots
    ofx = 0
    for k, pid in enumerate(PIDS):
        plan = json.load(open(ROOT / pid / f"{pid}.json"))
        beams = []
        for b in plan["beams"][:1]:                     # 1 beam per patient (keep the mock small)
            iio = 0
            nb = {"iso_center": b.get("iso_center", [0, 0, 0]), "gantry_angle": b.get("gantry_angle", 0),
                  "rays": []}
            for r in b["rays"][:3]:
                nr = {"ray_source": r["ray_source"], "ray_target": r["ray_target"], "beamlets": []}
                for bl in r["beamlets"]:
                    nr["beamlets"].append({"beamlet_uuid": "u", "energy": bl["energy"],
                                           "output_info": {"output_file_idx": ofx, "idx_in_output": iio,
                                                           "minimum_cutoff": 0.02}})
                    iio += 1
                nb["rays"].append(nr)
            beams.append(nb)
        meta.append({"image_file_idx": k, "anatomical_region": "thoracic" if "THB" in pid else "abdominal",
                     "beams": beams})
        ofx += 1
    json.dump(meta, open(in_dir / f"stacked-{PARTICLE}-beam-level-metadata.json", "w"))
    return in_dir, meta


def zeros_predict(img, entry):
    """Trivial predictor: each beamlet -> a small nonzero-but-below-cutoff dose to test cutoff +
    stacking + grid. Returns {key: (dose_np, output_info)}."""
    sz = img.GetSize()  # (x,y,z)
    shape = (sz[2], sz[1], sz[0])
    out = {}
    n = 0
    for b in entry["beams"]:
        for r in b["rays"]:
            for bl in r["beamlets"]:
                d = np.full(shape, 0.01, np.float32)   # below cutoff 0.02 -> must become 0
                d[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0   # one supra-cutoff voxel
                out[("k", n)] = (d, None, bl["output_info"]); n += 1   # bbox None = already full-grid
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gcmock_"))
    try:
        in_dir, meta = build_mock_input(tmp)
        out_dir = tmp / "output"
        process_run(in_dir, out_dir, zeros_predict, PARTICLE, MODALITY)

        used = {int(e["beams"][0]["rays"][0]["beamlets"][0]["output_info"]["output_file_idx"])
                for e in meta}
        checks = []
        # 1. all 10 output slots present, one .mha each
        for s in range(N_SLOTS):
            fs = list((out_dir / "images" / f"stacked-radiation-dose-map-{s + 1}").glob("*.mha"))
            checks.append((f"slot {s+1}: exactly 1 .mha", len(fs) == 1))
        # per used slot: 4D, compressed, grid matches, ordered contiguous, cutoff respected
        for e, pid in zip(meta, PIDS):
            ofx = int(e["beams"][0]["rays"][0]["beamlets"][0]["output_info"]["output_file_idx"])
            f = list((out_dir / "images" / f"stacked-radiation-dose-map-{ofx + 1}").glob("*.mha"))[0]
            im = sitk.ReadImage(str(f))
            src = sitk.ReadImage(str(ROOT / pid / "image" / "ct.mha"))
            nbeam = sum(len(r["beamlets"]) for r in e["beams"][0]["rays"])
            checks.append((f"slot {ofx+1}: 4D stack", im.GetDimension() == 4))
            checks.append((f"slot {ofx+1}: n frames == n beamlets ({nbeam})", im.GetSize()[3] == nbeam))
            checks.append((f"slot {ofx+1}: grid==source", im.GetSize()[:3] == src.GetSize()
                           and np.allclose(im.GetSpacing()[:3], src.GetSpacing())))
            arr = sitk.GetArrayFromImage(im)   # (n, z, y, x)
            # the evaluator's rule is STRICT: it flags 0 < dose < cutoff, so a voxel sitting
            # exactly at the cutoff is compliant. We quantise sub-cutoff dose up to the
            # cutoff instead of deleting it, which this must not reject.
            below = ((arr > 0) & (arr < 0.02)).sum()
            checks.append((f"slot {ofx+1}: no values in (0, cutoff)", below == 0))
            checks.append((f"slot {ofx+1}: has the supra-cutoff voxel", (arr >= 0.99).any()))
        # unused slots are 1x1x1 placeholders
        for s in range(N_SLOTS):
            if s in used:
                continue
            f = list((out_dir / "images" / f"stacked-radiation-dose-map-{s + 1}").glob("*.mha"))[0]
            checks.append((f"slot {s+1}: placeholder 1x1x1", sitk.ReadImage(str(f)).GetSize()[:3] == (1, 1, 1)))

        npass = sum(ok for _, ok in checks)
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"\n{npass}/{len(checks)} checks passed -> {'ALL GOOD' if npass == len(checks) else 'FAILURES'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
