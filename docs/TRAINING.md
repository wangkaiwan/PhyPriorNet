# Training Guide

End-to-end recipes to reproduce every released model. All hyperparameters live in YAML configs
under `configs/`; no training script hardcodes them. Paths inside the configs point to our local
layout; edit `cache_dir`, `splits`, and `run_root` for your machine.

## 0. Environment and data

- Python 3.11, PyTorch >= 2.10 (developed on 2.12) with CUDA, plus: `SimpleITK`, `numpy`,
  `pyyaml`, `pymedphys` (gamma), `wandb` (optional).
- Official DoseRAD2026 training data, laid out as released:
  `<ROOT>/{photon,proton}/training/<patient>/{image/{ct.mha,mr.mha}, dose/, <patient>.json}` and
  `<ROOT>/beam_parameters.json`.
- Single GPU (>= 24 GB recommended for base-48 training).
- Splits: we train the released models on **all 75 patients** (`splits_all75.json`: everything in
  `train`); the 5-fold CV protocol in the reports uses per-fold JSON splits generated once and
  reused everywhere (never random splits inside training scripts).

## 1. Photon-CT

```bash
# 1) Precompute the per-control-point channel cache (channels + dose crops at margin 24)
python scripts/precompute_photon_crops_skinentry.py            # see script header for arguments

# 2) Train the base-48 network (~120k steps), then finetune
python scripts/train.py --config configs/experiments/all75/all75_p1_photonct.yaml
python scripts/train.py --config configs/experiments/all75/all75_p2_ftg.yaml

# 3) (fast version) Distill to base-32 with the D-recipe:
#    stage 1: 200k steps of L1(GT) + 0.5*L1(teacher); stage 2: 40k GT-only finetune
python scripts/distill_dose_photon.py --config configs/experiments/all75/distill_photonct_b32_from4018.yaml
python scripts/distill_dose_photon.py --config configs/experiments/all75/distill_photonct_b32_from4018_Dft.yaml
```

Key config semantics: `distill_dose_photon.py` reads `init_student:`/`teacher_ckpt:`;
`train.py` reads `init_from:`. Unknown YAML keys are silently ignored, so use the right key for
the right script and verify the warm start from the first logged loss (a warm start begins near
convergence; a cold start begins around 0.3).

## 2. Photon-MRI

```bash
# 1) Pretrain the MR tissue classifier (with strong intensity-shift augmentation) and refiner
python scripts/train_sct_classifier.py ...    # see script header; the released clf is "clf_whole_mraug"
python scripts/train_sct_refiner.py ...

# 2) Joint dose-aware training (synthesizer + dose net, warm-started from Photon-CT)
python scripts/train_dose_e2e.py --config configs/experiments/all75/all75_p3_photonmri.yaml
python scripts/train_dose_e2e.py --config configs/experiments/all75/all75_p4_mmB.yaml   # multi-modal round

# 3) (fast version) Domain-adapt the base-32 student to synthesis-domain channels:
python scripts/precompute_photon_crops_sct_m24.py   # channels on the deployed front end's densities
python scripts/distill_dose_photon.py --config configs/experiments/all75/distill_photonmri_b32_sctft.yaml
# then stitch the student into the E2E checkpoint (synth.* from the deployed E2E + dose.* from the student)
```

The stitch is a plain state-dict merge; the deployed front end (synthesizer + classifier) must be
the same one that generated the training cache.

## 3. Proton-CT

```bash
# 1) Precompute WEPL + pencil-beam prior caches
python scripts/precompute_proton.py
python scripts/precompute_proton_prior.py

# 2) Train base-48 (~120k) and finetune on the GPU pencil-beam prior (train = deploy)
python scripts/train_dose_proton.py --config configs/experiments/all75/all75_r1_protonct.yaml
python scripts/train_dose_proton.py --config configs/experiments/all75/all75_r2_ft.yaml

# 3) (fast version) Convention-consistency finetune for the batched engine
#    (the orthographic per-beam WEPL differs from the per-voxel march by ~0.18 g/cm^2 mean abs)
python scripts/train_dose_proton.py --config configs/experiments/all75/all75_r2_ft_v3physics.yaml
```

The batched engine itself is `accel/proton_engine_v2.py`, enabled at deploy time with
`DOSERAD_ENGINE_V2=1` (see the fast bundle's `DEPLOY_ENV.txt` for the full knob set).

## 4. Proton-MRI

```bash
# 1) Classifier/refiner as in Photon-MRI (shared front end; classifier must run sliding-window
#    at the native grid in deployment)
# 2) Joint dose-aware E2E with density-direct synthesis and the WEPL-consistency loss
python scripts/train_dose_proton_e2e.py --config configs/experiments/all75/all75_r3_protonmri.yaml
# 3) Shift-robust finetune (the single largest held-out improvement of our campaign)
python scripts/train_dose_proton_e2e.py --config configs/experiments/all75/all75_r3ft2_mraug_protonmri.yaml
```

## 5. Evaluation

`scripts/eval_official_held16.py` scores a container-equivalent pipeline on a frozen 16-patient
cohort with our byte-verified reimplementation of the official metrics (per-beam masked MAE, IDD,
stratified MAE, local gamma 1%/1mm with the >= 10% Rx cutoff). We gated every deployment change
on this harness under one rule: a change ships only if it loses neither accuracy nor speed. Two
hard-won cautions: score photon models against the FULL-GRID ground truth (cropped scoring hides
scatter-tail loss), and never conclude from in-sample numbers (three of our platform submissions
scored 0.3 to 1.5 gamma below their in-sample parity).
