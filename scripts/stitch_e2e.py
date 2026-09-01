#!/usr/bin/env python3
"""Merge a separately trained dose network into an end-to-end (synthesizer + dose) checkpoint.

The MRI systems are one checkpoint holding `synth.*` (classifier/refiner) and `dose.*` (the dose
network). To deploy a distilled dose network behind the existing front end, we keep `synth.*`
from the end-to-end model and replace `dose.*` with the student's weights.

    python scripts/stitch_e2e.py --e2e $WORKDIR/runs/m24S2_p4_mmB/state.pt \
        --dose $WORKDIR/runs/distill_photonmri_b32_sctft/state.pt \
        --out $WORKDIR/runs/photonmri_b32_stitched.pt

The student must have been trained on channels produced by this same front end (see
precompute_photon_crops_sct_m24.py); otherwise it sees an input distribution it never saw in
training, which cost us 4 gamma points when we tried it.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import os

import torch


def _sd(ckpt):
    for key in ("ema", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default=os.path.expandvars("$WORKDIR/runs/m24S2_p4_mmB/state.pt"),
                    help="end-to-end checkpoint providing synth.*")
    ap.add_argument("--dose", default=os.path.expandvars(
        "$WORKDIR/runs/distill_photonmri_b32_sctft/state.pt"), help="dose network to graft in")
    ap.add_argument("--out", default=os.path.expandvars("$WORKDIR/runs/photonmri_b32_stitched.pt"))
    a = ap.parse_args()

    e2e = _sd(torch.load(a.e2e, map_location="cpu"))
    dose = _sd(torch.load(a.dose, map_location="cpu"))

    synth = {k: v for k, v in e2e.items() if k.startswith("synth.")}
    if not synth:
        raise SystemExit(f"no synth.* keys in {a.e2e}")
    old_dose = {k[len("dose."):] for k in e2e if k.startswith("dose.")}
    if old_dose and set(dose) != old_dose:
        only_new = sorted(set(dose) - old_dose)[:3]
        only_old = sorted(old_dose - set(dose))[:3]
        print(f"[warn] dose key sets differ (new-only e.g. {only_new}, old-only e.g. {only_old}); "
              f"this is expected when the student has a different width")

    merged = dict(synth)
    merged.update({f"dose.{k}": v for k, v in dose.items()})
    torch.save({"ema": merged}, a.out)
    n_syn = sum(v.numel() for v in synth.values()) / 1e6
    n_dose = sum(v.numel() for v in dose.values()) / 1e6
    print(f"wrote {a.out}: synth {n_syn:.2f} M + dose {n_dose:.2f} M params ({len(merged)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
