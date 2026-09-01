# PhyPriorNet

**A unified differentiable physics-prior residual 3D U-Net for beam-level dose prediction across
photon/proton × CT/MRI.** Developed for the DoseRAD2026 Grand Challenge (all four tasks) by
AMC_DoseCalc: Kai Wang (corresponding), Meixu Chen, Rui Yang — Department of Radiation Oncology,
University of Colorado Anschutz Medical Campus.

> Status: private until the challenge's public-release date.
> **Model weights are available upon request**: contact Kai Wang (kai.2.wang@cuanschutz.edu).
> See [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) for the released versions and their metrics.

## The challenge

| Resource | Link |
|---|---|
| Challenge website | https://doserad2026.grand-challenge.org |
| Final submission requirements | https://doserad2026.grand-challenge.org/final-submission-requirements/ |
| Official code and evaluation repositories | https://github.com/orgs/DoseRAD2026/repositories |
| Dataset download (Zenodo, 864 GB; HuggingFace mirror linked there) | https://doi.org/10.5281/zenodo.19347848 |
| Dataset paper | https://doi.org/10.48550/arXiv.2604.12778 |
| Underlying CT-MR cohort (SynthRAD2025) | https://doi.org/10.1002/mp.17981 |
| Ground-truth Monte-Carlo simulation code (Geant4) | https://github.com/DoseRAD2026/geant4-dose-sim |

Beyond the released data, the organizers' Geant4 simulation code (linked above) lets you
generate beam-level Monte-Carlo dose on **your own** CT/MR cohort and beam model; our pipeline
consumes exactly that layout, so it can be retrained on institutional data without modification
(this is also how one would tailor the analytical prior to a local machine, see the reports'
future-work sections).

The public training set used here is 75 patients (39 thorax, 36 abdomen) with paired planning CT,
0.35 T MR, beam parameters, and per-beam Geant4 Monte-Carlo dose. **The challenge dataset is
released under CC BY-NC 4.0 (non-commercial); this repository distributes code and model weights
only, no challenge data, and any use of the data remains subject to the challenge licence.**

## Method in one paragraph

For each beam element (photon MLC control point or proton beamlet), an analytical,
**differentiable** physics operator turns a density image and the beam descriptor into
dose-shaped input channels — radiological depth and MLC-projected fluence for photons;
skin-entry WEPL and a self-contained GPU Hong pencil-beam prior (~95% of the dose) for protons.
A compact 3D U-Net (DoseUNet3D: ASPP bottleneck, Softplus head, FiLM modality conditioning)
learns only the **residual to Monte Carlo**. On CT the density comes from the HU calibration; on
MRI a classifier→refiner synthesizer produces it and is trained **jointly through the physics**
with the dose loss, so the synthetic CT is optimized for dose, not image fidelity (and we show
image fidelity does not predict dose accuracy). Deployment adds knowledge-distilled students,
a batched proton engine (per-beam orthographic BEV cumulative-sum WEPL, 388× on the WEPL stage),
and streaming writers — every speed lever gated to lose no accuracy.

## Repository layout

```
run_task.py   one-command training per task (chains all stages)
examples/    smoke_test.py: synthetic-phantom installation check (CPU, no data needed)
doserad/     core package: differentiable physics operators, DoseUNet3D, data pipeline
accel/       deployment acceleration: batched proton engine v2, GPU channel builders
container/   the four Grand-Challenge invoke containers (photon/proton × CT/MRI)
scripts/     the 14 scripts needed to reproduce the released models
configs/released/  the 12 configs of the released models (paths as ${DATA_ROOT}/${WORKDIR})
archive/     the full development campaign (~140 scripts, ~190 configs), kept for provenance
reports/     the four per-task LNCS method reports (PDF + source)
docs/        WHICH_FILE.md · MODEL_ZOO.md · TRAINING.md · INFERENCE.md
```

## Installation

```bash
git clone https://github.com/wangkaiwan/PhyPriorNet && cd PhyPriorNet
# install PyTorch matched to your CUDA first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt      # or: pip install -e .[eval]
```

Python >= 3.10, one CUDA GPU (>= 24 GB to train the base-48 networks; the released fast models
run in ~5 GB). Scripts can be run from anywhere; they add the repository root to `sys.path`
themselves.

Verify the installation in one minute, without challenge data or a GPU:

```bash
python examples/smoke_test.py     # synthetic phantom -> physics channels -> network forward
```

## Reproducing our results

0. **The short path**: one command trains a whole task end to end, chaining every stage with the
   released configs (identical to running them by hand):

   ```bash
   export DATA_ROOT=/path/to/DoseRad2026_raw WORKDIR=/path/to/workdir
   python run_task.py photon_ct --dry-run     # see the plan first
   python run_task.py photon_ct               # quality model
   python run_task.py photon_ct --with-fast   # plus the distilled fast model
   ```

   Completed stages are skipped on re-run, and a failed stage prints the exact
   `--from-stage N` command to resume. Tasks: `photon_ct`, `photon_mri`, `proton_ct`, `proton_mri`.

1. **Inference with released weights**: [docs/INFERENCE.md](docs/INFERENCE.md) — download a
   bundle, build the container, run on the Grand-Challenge job layout. Bundles were extracted
   from the exact Docker images that produced our leaderboard entries.
2. **Which file do I run?**: [docs/WHICH_FILE.md](docs/WHICH_FILE.md) — what each of the 14
   scripts and 12 configs does, in run order, with a minimal single-task walkthrough.
3. **Training from scratch**: [docs/TRAINING.md](docs/TRAINING.md) — per-task recipes
   (cache precompute → base training → finetunes → distillation), with the pitfalls we hit
   documented inline.
4. **Which model to use**: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) — per task, the
   highest-quality version and the fastest quality-gated version, with hidden-test metrics.

## Results (DoseRAD2026 preliminary hidden test set)

| Task | γ 1%/1mm (quality / fast) | Runtime (quality / fast) |
|---|---|---|
| Photon-CT | 95.51 / 93.87 | 103.6 s / 45.4 s |
| Photon-MRI | 84.11 / 82.90 | 94.8 s / 57.4 s |
| Proton-CT | 95.60 / 95.31 | 191.7 s / 101.1 s |
| Proton-MRI | 79.34 / 79.08 | 188.8 s / 110 s |

Internal 5-fold cross-validation on the 75 public patients: 93.4 / 91.1 / 96.7 / 87.4
(photon-CT / photon-MRI / proton-CT / proton-MRI); details, ablations, and the mechanistic
analyses (real-CT ceiling, range-error decomposition, fidelity non-predictiveness) are in the
per-task reports under `reports/`.

## Licence

Code in this repository is licensed under the **GNU General Public License v3.0** (see
[`LICENSE`](LICENSE)): free to use, modify and redistribute, provided derivative works are
released under the same licence. For use under different terms, contact the corresponding author.

Model weights are distributed **on request for non-commercial research use**
(kai.2.wang@cuanschutz.edu); commercial use of the weights requires a separate agreement. They
were trained on the DoseRAD2026 dataset, which is released under CC BY-NC 4.0; the dataset
licence governs any use of the data, and this repository contains no challenge data.

## Citation

Until a preprint is up, please cite the challenge reports:

```
Wang K, Chen M, Yang R. A Unified Differentiable Physics-Prior Residual 3D U-Net for Beam-Level
Dose Prediction across Photon and Proton Radiotherapy on CT and MRI. DoseRAD2026 Grand Challenge
method reports, 2026. https://github.com/wangkaiwan/PhyPriorNet
```

## Acknowledgments

Built on the DoseRAD2026 Grand Challenge dataset (SynthRAD2025 cohort); ground-truth dose by
Geant4 Monte Carlo; treatment plans generated with matRad. Large-language-model assistance
(Claude, Anthropic) was used for drafting documentation and report text; all methods,
experiments, analyses and conclusions are the authors' own.
