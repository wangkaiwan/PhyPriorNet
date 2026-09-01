#!/usr/bin/env python3
"""One-command training for a whole task.

The full recipe of any task is a chain of stages (cache precompute -> base training -> finetunes
-> optional distillation). Running them by hand is error-prone, so this wrapper executes the
same stages, in the right order, with the released configs. It is a thin dispatcher: it calls the
very scripts documented in docs/WHICH_FILE.md, so results are identical to running them yourself.

    export DATA_ROOT=/path/to/DoseRad2026_raw WORKDIR=/path/to/workdir
    python run_task.py photon_ct                  # quality model (default)
    python run_task.py photon_ct --with-fast      # also train the distilled fast model
    python run_task.py proton_mri --dry-run       # print the plan, run nothing
    python run_task.py photon_ct --from-stage 2   # resume the chain at stage 2

Caches are the expensive part (hours, hundreds of GB). Stages that find their output already
present are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
CFG = "configs/released"

# stage := (label, argv, output-marker relative to WORKDIR or None)
TASKS = {
    "photon_ct": {
        "quality": [
            ("splits", ["scripts/build_splits.py", "--all75"], "splits_all75.json"),
            ("cache: photon channels (margin 24)", ["scripts/precompute_photon_crops_skinentry.py"],
             "cache/crops/photon_skinentry_m24"),
            ("train: base-48 from scratch", ["scripts/train.py", "--config", f"{CFG}/all75_p1_photonct.yaml"],
             "runs/all75_p1_photonct/state.pt"),
            ("train: finetune (submitted quality model)",
             ["scripts/train.py", "--config", f"{CFG}/all75_p2_ftg.yaml"], "runs/all75_p2_ftg/state.pt"),
        ],
        "fast": [
            ("distil: base-32 student (200k)",
             ["scripts/distill_dose_photon.py", "--config", f"{CFG}/distill_photonct_b32_from4018.yaml"],
             "runs/distill_photonct_b32_from4018/state.pt"),
            ("distil: GT-only finetune (40k, submitted fast model)",
             ["scripts/distill_dose_photon.py", "--config", f"{CFG}/distill_photonct_b32_from4018_Dft.yaml"],
             "runs/distill_photonct_b32_from4018_Dft/state.pt"),
        ],
    },
    "photon_mri": {
        "quality": [
            ("splits", ["scripts/build_splits.py", "--all75"], "splits_all75.json"),
            ("cache: photon channels (margin 24)", ["scripts/precompute_photon_crops_skinentry.py"],
             "cache/crops/photon_skinentry_m24"),
            ("train: MR tissue classifier (shift augmentation)", ["scripts/train_sct_classifier.py"],
             "sct_runs/clf_whole_mraug/best.pt"),
            ("cache: coarse CT from MR", ["scripts/precompute_coarse_ct.py"], "cache/coarse_ct_whole_soft"),
            ("train: sCT refiner", ["scripts/train_sct_refiner.py"], "sct_runs/refiner/best.pt"),
            ("train: dose-aware joint stage",
             ["scripts/train_dose_e2e.py", "--config", f"{CFG}/m24S2_p3_photonmri.yaml"],
             "runs/m24S2_p3_photonmri/state.pt"),
            ("train: multi-modal round (submitted quality model)",
             ["scripts/train_dose_e2e.py", "--config", f"{CFG}/m24S2_p4_mmB.yaml"],
             "runs/m24S2_p4_mmB/state.pt"),
        ],
        "fast": [
            ("cache: channels on synthesized densities",
             ["scripts/precompute_photon_crops_sct_m24.py"], "cache/crops/photon_skinentry_sct_m24"),
            ("distil: domain-adapt the base-32 student",
             ["scripts/distill_dose_photon.py", "--config", f"{CFG}/distill_photonmri_b32_sctft.yaml"],
             "runs/distill_photonmri_b32_sctft/state.pt"),
            ("stitch: synth.* (quality model) + dose.* (student) -> one checkpoint",
             ["scripts/stitch_e2e.py"], "runs/photonmri_b32_stitched.pt"),
        ],
    },
    "proton_ct": {
        "quality": [
            ("splits", ["scripts/build_splits.py", "--all75"], "splits_all75.json"),
            ("cache: proton WEPL + dose crops", ["scripts/precompute_proton.py"], "cache/crops/proton"),
            ("cache: GPU pencil-beam prior", ["scripts/precompute_proton_prior.py"], "cache/crops/proton_prior"),
            ("train: base-48 from scratch",
             ["scripts/train_dose_proton.py", "--config", f"{CFG}/all75_r1_protonct.yaml"],
             "runs/all75_r1_protonct/state.pt"),
            ("train: finetune on the GPU prior (submitted quality model)",
             ["scripts/train_dose_proton.py", "--config", f"{CFG}/all75_r2_ft.yaml"],
             "runs/all75_r2_ft/state.pt"),
        ],
        "fast": [
            ("train: convention-consistency finetune for the batched engine",
             ["scripts/train_dose_proton.py", "--config", f"{CFG}/all75_r2_ft_v3physics.yaml"],
             "runs/all75_r2_ft_v3physics/state.pt"),
        ],
    },
    "proton_mri": {
        "quality": [
            ("splits", ["scripts/build_splits.py", "--all75"], "splits_all75.json"),
            ("cache: proton WEPL + dose crops", ["scripts/precompute_proton.py"], "cache/crops/proton"),
            ("cache: GPU pencil-beam prior", ["scripts/precompute_proton_prior.py"], "cache/crops/proton_prior"),
            ("train: MR tissue classifier (shift augmentation)", ["scripts/train_sct_classifier.py"],
             "sct_runs/clf_whole_mraug/best.pt"),
            ("cache: coarse CT from MR", ["scripts/precompute_coarse_ct.py"], "cache/coarse_ct_whole_soft"),
            ("train: sCT refiner", ["scripts/train_sct_refiner.py"], "sct_runs/refiner/best.pt"),
            ("train: dose-aware end-to-end (density-direct + WEPL consistency)",
             ["scripts/train_dose_proton_e2e.py", "--config", f"{CFG}/all75_r3_protonmri.yaml"],
             "runs/all75_r3_protonmri/state.pt"),
            ("train: shift-robust finetune (submitted model)",
             ["scripts/train_dose_proton_e2e.py", "--config", f"{CFG}/all75_r3ft2_mraug_protonmri.yaml"],
             "runs/all75_r3ft2_mraug_protonmri/state.pt"),
        ],
        "fast": [],   # same weights as the quality model; speed comes from the deploy engine
    },
}

FAST_NOTE = {
    "proton_mri": "The fast Proton-MRI submission reuses the quality weights; its speed comes "
                  "from the deployment engine (DOSERAD_ENGINE_V2=1), not from separate training.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--with-fast", action="store_true", help="also train the fast (distilled/engine) model")
    ap.add_argument("--only-fast", action="store_true", help="skip the quality chain")
    ap.add_argument("--from-stage", type=int, default=1, help="1-based stage to start from")
    ap.add_argument("--force", action="store_true", help="run stages even if their output exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    data_root, workdir = os.environ.get("DATA_ROOT"), os.environ.get("WORKDIR")
    if not (data_root and workdir):
        print("Set DATA_ROOT (challenge data) and WORKDIR (scratch space) first; "
              "see docs/WHICH_FILE.md", file=sys.stderr)
        return 2

    stages = []
    if not args.only_fast:
        stages += TASKS[args.task]["quality"]
    if args.with_fast or args.only_fast:
        stages += TASKS[args.task]["fast"]
        if not TASKS[args.task]["fast"] and args.task in FAST_NOTE:
            print(f"[note] {FAST_NOTE[args.task]}")

    print(f"\n{args.task}: {len(stages)} stage(s)\n" + "-" * 60)
    for i, (label, argv, marker) in enumerate(stages, 1):
        done = marker and (Path(workdir) / marker).exists()
        flag = "  [output exists, will skip]" if done and not args.force else ""
        print(f"{i:>2}. {label}{flag}")
    print("-" * 60)
    if args.dry_run:
        return 0

    for i, (label, argv, marker) in enumerate(stages, 1):
        if i < args.from_stage:
            continue
        if marker and (Path(workdir) / marker).exists() and not args.force:
            print(f"\n[{i}/{len(stages)}] SKIP (exists): {label}")
            continue
        print(f"\n[{i}/{len(stages)}] {label}\n    $ python {' '.join(argv)}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable] + argv, cwd=REPO)
        if rc != 0:
            print(f"\nStage {i} failed (exit {rc}): {label}\n"
                  f"Fix the cause and resume with:  python run_task.py {args.task} --from-stage {i}",
                  file=sys.stderr)
            return rc
        print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\nAll stages complete. Evaluate with:\n"
          f"    python scripts/eval_official_held16.py {args.task}\n"
          f"Deploy with the container of this task (see docs/INFERENCE.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
