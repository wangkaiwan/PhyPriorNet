# Model Zoo

Two released versions per task: the **highest-quality** model and the **fastest quality-acceptable**
model. All metrics below are from the DoseRAD2026 preliminary hidden test set (August 2026
re-scored metrics); runtime is the total per-job time on the challenge evaluation instance.
Weight bundles are attached to the GitHub Release `v1.0-doserad2026`; each bundle contains the
network weights, the matching deploy config, `beam_parameters.json`, and `DEPLOY_ENV.txt` (the
exact environment variables baked into the scored container). Weights were extracted from the
exact Docker images that produced the leaderboard entries (image IDs in the release notes).

| Task | Version | Model | gamma 1%/1mm | Runtime | Bundle |
|---|---|---|---|---|---|
| Photon-CT | quality | base-48 DoseUNet3D, margin 24, cutoff-zeroed | **95.51** | 103.6 s | `photonct_quality.tar.gz` |
| Photon-CT | fast | distilled base-32 (D-recipe), margin 16 | 93.87 | **45.4 s** | `photonct_fast.tar.gz` |
| Photon-MRI | quality | mm-trained base-48 E2E + shift-aug classifier, margin 16 | **84.11** | 94.8 s | `photonmri_quality.tar.gz` |
| Photon-MRI | fast | sCT-domain-adapted base-32 student, same front end | 82.90 | **57.4 s** | `photonmri_fast.tar.gz` |
| Proton-CT | quality | base-48 + GPU pencil-beam prior (per-voxel WEPL engine) | **95.60** | 191.7 s | `protonct_quality.tar.gz` |
| Proton-CT | fast | batched engine v2 (BEV cumulative-sum WEPL, bucketed batch) | 95.31 | **101.1 s** | `protonct_fast.tar.gz` |
| Proton-MRI | quality | dose-aware synth + base-48, shift-aug classifier | **79.34** | 188.8 s | `protonmri_quality.tar.gz` |
| Proton-MRI | fast | same accuracy chain on the batched engine v2 | 79.08 | **110 s** | `protonmri_fast.tar.gz` |

Notes:
- "Fast" versions were selected under a strict gate during development: a speed change shipped
  only if the accuracy cost was small and measured (see the per-task reports in `reports/`).
- The proton fast engine is bit-compatible with the quality network family but computes per-beam
  WEPL with an orthographic beam's-eye-view cumulative sum (beamlets of one beam are exactly
  parallel), batch-forwards shape-bucketed crops, and streams compressed output.
- To reproduce a leaderboard container exactly: place the bundle's files into
  `container/<task>/weights/`, apply `DEPLOY_ENV.txt`, and build with `container/<task>/build.sh`
  (see `docs/INFERENCE.md`).
