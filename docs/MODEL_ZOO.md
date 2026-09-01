# Model Zoo

Two released versions per task: the **highest-quality** model and the **fastest quality-acceptable**
model. All metrics below are from the DoseRAD2026 preliminary hidden test set (August 2026
re-scored metrics).

**What the runtime column means.** It is the wall-clock time of one scored evaluation job on the
challenge instance: model load, the per-beam-element loop, and writing every dose map to disk.
A job is a batch of beam elements, not one patient, and its size varies with the test set. In our
scored jobs a photon job contained on the order of 500 control points and a proton job 177 to 309
beamlet dose maps. The useful invariant is therefore the per-element cost, which we also give:

| Task | version | per beam element | example job | what that job is |
|---|---|---|---|---|
| Photon-CT | quality | ~0.19 s / control point | 103.6 s | ~540 control points, i.e. one full VMAT plan (two arcs at ~180 control points per arc, plus the fixed overhead) |
| Photon-CT | fast | ~0.08 s / control point | 45.4 s | same ~540 control points |
| Photon-MRI | quality | ~0.17 s / control point | 94.8 s | ~540 control points plus ~2 s of one-off MR-to-density synthesis per patient |
| Photon-MRI | fast | ~0.10 s / control point | 57.4 s | same, with the distilled network |
| Proton-CT | quality | ~0.62 s / beamlet dose map | 191.7 s | 309 beamlet dose maps (measured: 62.8 s for 309 maps with the fast engine, 203 ms each) |
| Proton-CT | fast | ~0.20 s / beamlet dose map | 101.1 s | 309 beamlet dose maps, ~27 GB of output written |
| Proton-MRI | quality | ~0.61 s / beamlet dose map | 188.8 s | 309 beamlet dose maps plus ~2 s synthesis |
| Proton-MRI | fast | ~0.36 s / beamlet dose map | 110 s | same, batched engine |

For clinical intuition: a photon control point is one MLC segment of a VMAT arc, so the fast
photon model computes a complete two-arc plan in under a minute, and a single segment in about
0.1 s, which is fast enough to recompute dose while a therapist reviews the image of the day.
A proton beamlet is one energy-and-spot element; clinical spot maps contain thousands of them, so
a full proton plan is several minutes at the per-element rates above, dominated by writing the
per-beamlet dose volumes on the native 1x1x3 mm grid rather than by the network.
Weight bundles are **available upon request** (contact kai.2.wang@cuanschutz.edu); each bundle contains the
network weights, the matching deploy config, `beam_parameters.json`, and `DEPLOY_ENV.txt` (the
exact environment variables baked into the scored container). Weights were extracted from the
exact Docker images that produced the leaderboard entries (image IDs available with the bundles).

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
