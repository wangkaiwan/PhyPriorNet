# Inference / Deployment Guide

Each task ships as a self-contained Docker container (Grand-Challenge invoke interface): an HTTP
app receives the job, loads the image and beam JSON, and writes one `.mha` dose map per beam
element. No external treatment-planning system is required at inference; the proton containers
compute their own pencil-beam prior on GPU.

## Quick start (per task)

```bash
# 1) Download a bundle from the GitHub Release and unpack it into the container's weights dir
tar -xzf photonct_quality.tar.gz -C container/photon/weights/

# 2) Apply the deployment environment of that version
#    (DEPLOY_ENV.txt in the bundle lists the exact variables of the scored container;
#     build.sh bakes the defaults, override at `docker run -e ...` for variants)

# 3) Build
bash container/photon/build.sh container/photon/weights/photon.pt
# photon-MRI / proton-MRI build.sh additionally take the classifier checkpoint and deploy config

# 4) Run (Grand-Challenge layout: /input holds the job, /output receives dose maps)
docker run --rm --gpus all -v <job>:/input -v <out>:/output doserad-photon:latest
```

## The deployment knobs that matter

| Env | Meaning | Guidance |
|---|---|---|
| `DOSERAD_PHOTON_MARGIN` | crop margin (voxels) around the MLC aperture | 24 = quality, 16 = fast; select on FULL-GRID metrics (cropped scoring hides the scatter tail) |
| `DOSERAD_CUTOFF_ZERO` | zero sub-cutoff output (vs quantise) | 1 for photon (matches official GT), 0/quantise for proton |
| `DOSERAD_COMPRESS_OUTPUT` | zlib output compression | 0 for photon (compute-bound), 1 for proton (write-bound, ~2x) |
| `DOSERAD_ENGINE_V2` | batched proton engine (BEV cumsum WEPL + bucketed batch) | fast proton versions; pair with `EV2_CHUNK`, `EV2_MAXVOX` (memory budget) |
| `DOSERAD_WRITE_POOL` | async writer threads | 4 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | required at 16 GB-class VRAM |

## Platform lessons (worth reading before you deploy anywhere)

- **torch.compile**: pays off for photon (one grid per patient, hundreds of same-shape control
  points amortize one compile) and is a net loss for proton on services that start a fresh
  process per job (a ~50 s compile tax per job, and multi-grid jobs recompile repeatedly).
- **Memory**: validate under the target GPU's budget, not your dev GPU's. Two of our submissions
  OOMed on the platform because a 32 GB dev GPU masked the limit; replaying the largest patients
  under a hard cap caught the fix.
- **Small nets saturate at batch 1**: for the distilled base-32 photon student, control-point
  batching, async emission, and fp16 weights all measured <= 1.09x or negative. The convolution
  forward is the floor once channels and writing are off the critical path.
- The proton batched engine is exact by construction where it matters: same-shape bucketing with
  plain GroupNorm keeps batching bit-identical; the WEPL convention change is absorbed by a
  short finetune (train = deploy).
