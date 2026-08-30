# PhyPriorNet

A unified differentiable physics-prior residual 3D U-Net for beam-level dose prediction across
photon/proton x CT/MRI. Developed for the DoseRAD2026 Grand Challenge (all four tasks) by
AMC_DoseCalc (Kai Wang, Meixu Chen, Rui Yang — University of Colorado Anschutz Medical Campus).

**Status: code release in preparation** (full implementation, differentiable physics operators,
GPU pencil-beam prior, deployment containers, and final weights will be added before 2026-10-01
per the challenge open-source requirement).

## Reports

Per-task LNCS method reports (final-submission versions) are in [`reports/`](reports/):

| Task | Report |
|---|---|
| Photon-CT | `reports/paper_LNCS_photonct_v1.pdf` |
| Photon-MRI | `reports/paper_LNCS_photonmri_v1.pdf` |
| Proton-CT | `reports/paper_LNCS_protonct_v1.pdf` |
| Proton-MRI | `reports/paper_LNCS_protonmri_v1.pdf` |

Method in one line: per beam element (photon MLC control point / proton beamlet), analytical
differentiable physics channels (radiological depth / WEPL / GPU Hong pencil-beam prior) feed a
residual 3D U-Net that corrects the prior toward Monte Carlo; on MRI a classifier-refiner
synthesizer is trained jointly through the physics, making the synthetic CT dose-aware.
