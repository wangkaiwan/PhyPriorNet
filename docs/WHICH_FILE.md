# Which script / config do I need?

The repository keeps only the files needed to reproduce the four submitted models in `scripts/`
and `configs/released/`. Everything else from the development campaign (about 140 scripts and
190 configs: ablations, probes, dead ends, one-off analyses) is preserved unedited under
`archive/` for provenance; you do not need any of it to reproduce our results.

## First: check the installation

```bash
python examples/smoke_test.py
```

Builds a synthetic phantom, runs the analytical physics operator and an untrained network on CPU,
and prints channel ranges. If this passes, your environment is fine and any later failure is
about data or configuration, not the install.

## Shortcut: one command per task

If you do not want to think about stage order at all:

```bash
export DATA_ROOT=/path/to/DoseRad2026_raw WORKDIR=/path/to/workdir
python run_task.py <task> [--with-fast] [--dry-run] [--from-stage N] [--force]
```

`run_task.py` is a thin dispatcher over the same scripts and configs listed below, so the result
is identical to running the stages yourself. It prints the plan, skips stages whose output
already exists, and on failure tells you how to resume. Start with `--dry-run` to see what a task
would cost (the cache stages are the expensive ones). The rest of this document explains what the
individual pieces do, which you need when you want to deviate from our recipe.

## Path placeholders

The released configs use two placeholders. Substitute them (or export them and expand with
`envsubst`) before running:

- `${DATA_ROOT}` — the official challenge data, e.g. `/data/DoseRad2026_raw`, containing
  `photon/training/<pid>/...`, `proton/training/<pid>/...`, and `beam_parameters.json`.
- `${WORKDIR}` — your scratch space for caches, splits and run outputs (needs ~1 TB if you build
  all caches; ~250 GB for a single task).

## The 15 scripts

| Script | What it does | Used by |
|---|---|---|
| `build_splits.py` | writes the patient-level split JSON (all-75 or k-fold) | all tasks, run first |
| `precompute_photon_crops_skinentry.py` | photon channel + dose crop cache (per control point) | photon-CT, photon-MRI |
| `precompute_photon_crops_sct_m24.py` | same channels but on synthesizer-derived densities | photon-MRI fast student only |
| `precompute_proton.py` | proton WEPL + dose crop cache (per beamlet) | proton-CT, proton-MRI |
| `precompute_proton_prior.py` | GPU Hong pencil-beam prior cache | proton-CT, proton-MRI |
| `precompute_coarse_ct.py` | tissue-class coarse CT from MR (synthesizer stage 1 output) | MRI tasks |
| `train.py` | photon dose-network trainer (CT task); reads `init_from:` | photon-CT |
| `train_dose_proton.py` | proton dose-network trainer (CT task) | proton-CT |
| `train_dose_e2e.py` | photon MR-to-dose end-to-end trainer (synthesizer + dose net) | photon-MRI |
| `train_dose_proton_e2e.py` | proton MR-to-dose end-to-end trainer | proton-MRI |
| `train_sct_classifier.py` | MR tissue classifier (use the intensity-shift augmentation flags) | MRI tasks |
| `train_sct_refiner.py` | sCT refiner pretraining (also uses public SynthRAD2025 pairs) | MRI tasks |
| `distill_dose_photon.py` | knowledge distillation; reads `init_student:` / `teacher_ckpt:` | fast photon models |
| `eval_official_held16.py` | our reimplementation of the official metrics on a fixed cohort | all tasks |
| `stitch_e2e.py` | merges a separately trained dose network into an end-to-end MRI checkpoint | photon-MRI fast model |

Trainer/config key pairing (a real bug we hit): `train.py` reads `init_from:`,
`distill_dose_photon.py` reads `init_student:`. Unknown YAML keys are silently ignored, so always
confirm a warm start from the first logged loss.

## The 14 released configs

Run them in the order listed; each row's output is the next row's starting point.

**Photon-CT** (quality model = step 2; fast model = steps 3-4)
1. `all75_p1_photonct.yaml` — base-48 dose network from scratch (~120k steps)
2. `all75_p2_ftg.yaml` — finetune (this is the submitted quality model)
3. `distill_photonct_b32_from4018.yaml` — distil to base-32 (200k, teacher + GT)
4. `distill_photonct_b32_from4018_Dft.yaml` — GT-only finetune of the student (40k)

**Photon-MRI** (quality model = step 2; fast model = step 3 stitched into step 2)
1. `m24S2_p3_photonmri.yaml` — dose-aware joint stage, warm-started from the photon-CT network
   plus the pretrained classifier/refiner
2. `m24S2_p4_mmB.yaml` — dose-aware joint training, multi-modal round (submitted quality model)
3. `distill_photonmri_b32_sctft.yaml` — domain-adapt the base-32 student on synthesis-domain
   channels, then merge `synth.*` (from step 2) with `dose.*` (the student) into one checkpoint
4. `m24S2_p4_mmB_b32dose.yaml` — the deploy-time config describing that merged network

**Proton-CT** (quality model = step 2; fast engine = step 3)
1. `all75_r1_protonct.yaml` — base-48 with WEPL + pencil-beam prior channels
2. `all75_r2_ft.yaml` — finetune on the GPU prior (train = deploy; submitted quality model)
3. `all75_r2_ft_v3physics.yaml` — convention-consistency finetune for the batched engine

**Proton-MRI**
1. `all75_r3_protonmri.yaml` — dose-aware end-to-end (density-direct output, WEPL-consistency loss)
2. `all75_r3ft2_mraug_protonmri.yaml` — shift-robust finetune (submitted model)

## Minimal path for a single task

Example, photon-CT quality model:

```bash
export DATA_ROOT=/path/to/DoseRad2026_raw WORKDIR=/path/to/workdir
python scripts/build_splits.py --all75                       # -> $WORKDIR/splits_all75.json
python scripts/precompute_photon_crops_skinentry.py          # -> $WORKDIR/cache/crops/... (~350 GB, margin 24)
python scripts/train.py --config configs/released/all75_p1_photonct.yaml
python scripts/train.py --config configs/released/all75_p2_ftg.yaml
python scripts/eval_official_held16.py photon_ct             # official-metric check
```

Most scripts take arguments (`--help` lists them); a few cache builders are driven purely by the
`DATA_ROOT` / `WORKDIR` environment variables and take none. Every script's module docstring
documents its inputs and outputs; read the header before running a multi-hour cache build.

## Training on your own data

The pipeline is not tied to the challenge cohort. The organizers publish the Geant4 simulation
code used to produce the ground truth (https://github.com/DoseRAD2026/geant4-dose-sim), so you
can generate beam-level Monte-Carlo dose for your own patients and beam model, lay it out in the
same directory structure (`<DATA_ROOT>/{photon,proton}/training/<pid>/{image/,dose/,<pid>.json}`
plus `beam_parameters.json`), and run the same recipes. For a local linac or proton machine you
will also want to re-fit the analytical prior to your commissioning data; the physics operators
in `doserad/physics/` are where the machine model enters.
