# Acceleration method matrix — target: max speed, gamma-lossless (<0.1 mean gamma delta)

Every method is verified on the SCORED metric (plan gamma 1%/1mm), not just dose rel-diff.
Speedups are per-component @RTX5090; A10G ~x2.5-3.5 slower (deploy target).

## 4-TASK END-TO-END DEPLOY (2026-07-18, container path, full plan 1ABB006, cold-inclusive, @5090)
Same methodology across all four -> directly comparable. All accel LOSSLESS (gamma unchanged).
| task       | element      | BEFORE      | AFTER (V6/V7) | speedup | accel applied            |
|------------|--------------|-------------|---------------|---------|--------------------------|
| Photon-CT  | ms/CP        | 131.3       | 89.4          | 1.47x   | V7 fast channels         |
| Photon-MRI | ms/CP        | 139         | 100           | 1.39x   | V7 fast channels         |
| Proton-CT  | ms/beamlet   | 84.6        | 49.2          | 1.72x   | V6 2-pass tight-WEPL     |
| Proton-MRI | ms/beamlet   | 90          | 52            | 1.73x   | V6 2-pass tight-WEPL     |
(A10G ~x2.5-3.5 slower; all gate-safe with margin. Photon net ~62ms is now the photon bottleneck.)

## VERIFIED WINS (bankable)
| # | method | task | component | speedup | accuracy | status |
|---|--------|------|-----------|---------|----------|--------|
| V1 | torch.compile(dynamic) net forward | photon | forward (78%) | 3.05x | gamma +0.009 (max 0.08) | ✅ locked |
| V2 | torch.compile(dynamic) net forward | proton | forward (20%) | 2.65x | gamma pending | ✅ (verify) |
| V3 | per-ray PB geometry sharing | proton | PB build (37%) | 2.01x | exact 5.7e-6 | ✅ locked |
| V4 | ray-centric build + FUSED WEPL (1 grid_sample -> src+skin WEPL+entered) | proton | build | **1.78x** e2e | gamma 98.7=98.7 (5 ch <2e-3) | ✅ locked |

## HIGH-VALUE TODO (designed, not yet built/measured)
| # | method | task | expected | risk |
|---|--------|------|----------|------|
| ~~T1~~ DONE=V4 | ray-centric build | proton | 3.24x full build | done, exact |
| T2 | cross-ray beamlet batching (concat voxels, one grid_sample) | proton | further build | med |
| T3 | proton net compile + gamma verify | proton | 2.65x forward (confirm) | low |
| T4 | CUDA streams: overlap build(beam i+1) with forward(beam i) | both | ~1.3-1.5x when balanced | med |
| T5 | photon channel build compile (rdepth/fluence raytrace) | photon | ? (22% of photon) | low |
| T6 | INT8 / TensorRT on the compiled net | both | ~1.5-2x on forward | med (QAT/calib) |
| T7 | distillation: base24/16 student (residual is small, esp. proton) | both | 3-5x forward + batchable | high (retrain) |
| T8 | FP16 weights + channels-last memory format | both | ~1.1-1.3x | low |
| T9 | lower WEPL march step (1mm->1.5/2mm) w/ gamma check | proton | ~1.5x WEPL | med (range accuracy) |

## MEASURED DEAD ENDS (don't revisit)
- padded-batch net forward (photon): memory-bound on ~6M-voxel crops -> 1.00x.
- masked-GN batching: correct but per-sample-loop overhead + big crops -> no win for photon.
- seq_fp32 (autocast off): slower. Autocast already deployed.
- scalar density calibration for proton range: signed R80 ~0 (unbiased) -> no fix.

## Principle
Compose multiplicatively: e.g. proton build (per-ray WEPL-share 4x + PB-share 2x) x forward (compile
2.65x) x streams overlap. Each step gamma-verified before stacking.

## Accel round 2026-07-18 (post-MRI-containers, profiling-driven)
- PROFILE (deploy proton, warm @5090, container/proton path): geom 0.4% | **build(WEPL) 66% (28ms)**
  | tight 0.6% | croppad 0.1% | net 31.8% (13.6ms) | xfer 1%. TOTAL ~42.7 ms/beamlet warm. The NET
  forward is small (2-5ms micro; ~13ms with autocast/guards) — the docs' "54ms net" was overhead.
  BUILD (fused WEPL grid_sample on the inflated geom-union box) is the real bottleneck.
- V5 channels_last_3d on the compiled net: 1.02-1.11x on the NET only, max|Δ| 5e-4 (lossless). But net
  is 32% -> saves ~0.5-1ms/beamlet. LOW VALUE, not worth the plumbing; REJECTED for now.
- T9 WEPL march step (LOSSY, gamma-verified abd 1ABB006 / lung 1THB002):
    1.0mm: abd 98.3 / lung 97.3, build ~63 ms/bl   |  1.5mm: 98.1 / 97.1 (BOTH -0.2), ~42 (1.49x)
    2.0mm: 97.6 / 96.7 (-0.6/-0.7), ~32 (1.96x). 1.5mm is the knee. AVAILABLE opt-in knob
  (build_ray wepl_step_mm) but COSTS gamma; prefer the lossless 2-pass. Gate already safe.
- **V6 2-PASS TIGHT-WEPL BUILD (accel/proton_2pass.py) — LOCKED, LOSSLESS, ~2x build.** The build
  runs one WEPL grid_sample over the geom-union box, but the net consumes only the PB-tight crop
  (ray-union geom/tight ratio: mean 4.48x, median 3.44x, min 1.88x). Pass 1: cheap coarse PB
  (stride-2 grid + 2mm step) locates each beamlet's tight bbox (conservative 0.005 thresh + margin).
  Pass 2: full-fidelity build_ray over that ~4.5x smaller union. Lossless because the net still sees
  the full-res channels on the SAME final _tight_from_pb crop (locate box is a superset). Margin sweep
  (lung 1THB002, binding case): m4 -0.1/2.9x, **m8 LOSSLESS/2.2x (LOCKED)**, m10 lossless/1.96x.
  Verified: abd 1ABB006 98.3=98.3, lung 97.3=97.3 (plan max|Δ| 1e-3 Rx), build 62->28 ms/bl. WIRED as
  container default (container/proton/predict.py, DOSERAD_2PASS=1; DOSERAD_2PASS_MARGIN=8). END-TO-END
  (proton-MRI container, all 1080 bl): 90 -> 52 ms/beamlet (1.73x), gamma 89.3 UNCHANGED. Benefits
  BOTH proton-CT + proton-MRI (shared predict_beams).
- **V7 PHOTON GPU-RESIDENT CHANNELS (accel/photon_channels_fast.py) — LOCKED, BIT-IDENTICAL, 6.2x on
  channels.** PROFILE (photon deploy, warm): channels 50.8ms (43%) + net 62ms (53%). ROOT of the
  channels cost: photon_channels re-runs `density.astype(float32)` (full 60MB CPU copy, ~30ms) +
  `torch.as_tensor(density,cuda)` (~6ms) EVERY CP, though density is constant/patient (measured
  36ms/CP waste); radiological_depth_fast re-uploads too (~6ms). Fix: upload density to GPU ONCE, pass
  the GPU tensor everywhere (rdepth accepts a tensor -> no re-upload). Channel MATH copied verbatim ->
  BIT-IDENTICAL (max|Δ|=0.0, bbox match over 10 CPs). Speed: photon_channels 50.8 -> 8.2 ms/CP (6.2x).
  WIRED into container/photon/predict.py (upload dens_t once) AND container/photon_mri/predict.py
  (also removes a density GPU->CPU->GPU round-trip). END-TO-END photon-MRI container: 139 -> 100 ms/CP
  (1.39x), gamma 91.3 UNCHANGED. Photon-CT even faster (no extra skinentry rdepth). Benefits BOTH
  photon-CT + photon-MRI (shared predict_cps). NOTE net (62ms) is now the photon bottleneck.
- POST-ACCEL PROFILE: proton-2pass warm 28.9 ms/bl = build 14.5 (50%) + net 13.5 (47%) — now balanced.
  photon warm = net 62 (compute-bound, large aperture crops mean 4.23M vox up to 6.09M) + channels 8.

## NET-LEVEL LEVERS — ALL EXHAUSTED (the dose-net compute is now the hard floor, 2026-07-18)
The net is the remaining bottleneck (proton ~13.5ms, photon ~62ms). No lossless win left with current
tooling — measured & rejected:
- channels_last_3d: proton net 1.02-1.11x BUT photon net 0.87-0.92x SLOWER (inductor default layout
  already optimal on large anisotropic crops; conversion overhead). REJECT.
- torch.compile(mode='reduce-overhead') / CUDA graphs: proton 1.05-1.14x (bit-identical), photon 1.01x
  (compute-bound). Marginal (~1.04x total) + CUDA-graph production risk w/ dynamic shapes (memory,
  fallback). REJECT for deploy.
- net batching (same-shape throughput CEILING): proton small crop 1.4x best (batch4) then plateaus;
  medium crop 1.08x. Base-48 3D UNet already utilizes the GPU at batch 1 -> the ceiling is low and
  would be eroded by pad-to-common + masked-GN overhead. Confirms docs' masked-GN dead end. REJECT.
- TensorRT / ONNX / INT8: torch_tensorrt/tensorrt/onnx NOT installed (torch 2.12+cu130); INT8 also
  gamma-risky. Distillation = retrain (out of scope). Both deferred.
NET RESULT of the accel round: Photon-CT 131->89, Photon-MRI 139->100, Proton-CT 85->49, Proton-MRI
90->52 ms/elem (all LOSSLESS). Further speed needs a lighter net (distillation) or TensorRT — both
non-trivial; all 4 tasks already gate-safe with margin, so this is a good stopping point.

## MORE dead ends (2026-07-17)
- photon raytrace fan resolution (n_u/n_v 128->96/64): rdepth changes 18-25% (would hurt gamma),
  only 1.43-1.46x. Rejected. Other O(volume) channel ops dominate, not the fan.

## CURRENT VERIFIED STATE (both tasks gate-safe with margin)
| task | per-elem @5090 | A10G est | vs 1s gate | accuracy |
|------|----------------|----------|-----------|----------|
| photon | ~120 ms/CP (compile 3x fwd + 55ms channels) | ~0.3-0.4 s/beam | PASS margin | gamma +0.009 |
| proton | 24.6 ms/beamlet (compile + fused ray-centric) | ~0.06-0.09 s/beam | PASS big margin | gamma 98.7=98.7 |

Remaining levers (diminishing/higher-effort, no clear low-risk win left):
CUDA streams (~1.3x, no accuracy risk, complex), INT8/TensorRT (med), distillation (retrain).

## FINAL DEPLOY STATE (2026-07-18 ~06:15) — real, gamma-verified, gate-safe
| task | deploy ms/elem @5090 (warm) | A10G est | gate | gamma (deploy, no GT) |
|------|------|------|------|------|
| proton | 73.7 ms/beamlet (lat 24mm, geom-bbox + PB-tight-crop + compiled net) | ~0.18-0.27 s/beam | PASS margin | abd 98.3 / lung 97.3 |
| photon | 115.9 ms/CP (aperture bbox + compiled net) | ~0.3-0.4 s/beam | PASS margin | 96.6 = per-patient baseline (lossless) |

NEXT ranking lever (bigger effort): batch the proton net forward across beamlets (tight crops are
small ~158k vox -> launch-bound; masked-GN for correctness) to cut the ~54ms net+overhead share.
Deploy gap vs GT-bbox 24.6ms is: build on geom box + net on PB-tight (bigger than GT) crops +
per-beamlet python/transfer overhead. Gate is already safe; this is pure ranking optimization.

## V8 — Distillation base32 (2026-07-19): TEACHER PARITY at 2.1x
Student DoseUNet3D base=32 (7.6M params) distilled 200k from ftg_skinentry_photonct_f0 (base48,
17.1M; loss = weighted_l1(GT) + 0.5*weighted_l1(teacher), batch2, OneCycle 2e-4).
Fold-0 plan gamma 1%/1mm (16 val pts): student 96.1 abd / 91.8 lung / 94.0 ALL == teacher
96.1/91.9/94.0. Trajectory 40k 77.7 -> 80k 86.9 -> 160k 93.2 -> 200k 94.0 (converged; no extension
needed). Net 2.10x -> photon-CT e2e 89.4 -> ~57 ms/CP (1.55x). Weights:
runs/distill_photonct_b32_f0/snap_200000.pt. Eval CSVs: cv_eval/distill_b32_f0_*.csv.
b24 (2.60x) probing next — same recipe, configs/experiments/cv/distill_photonct_b24_f0.yaml.

## V9 — b32-era photon per-CP micro-levers (2026-08-27): ALL DEAD, floor reached
Final-candidate = b32 D-ft @ m16 (35.3 ms/CP warm incl. writer; board proj ~49s). Tried and killed:
- HALF=1 (net.half): 35.3 -> 35.3, zero gain (autocast already fp16) + numeric risk -> NO.
- compile max-autotune: 34.7 (-1.7%, noise) and platform t_fix pays the longer autotune -> NO.
- CP batching (b32, bucketed): 1.01-1.09x at batch4/8 — b32 STILL saturates at batch1, same
  verdict as the base48-era padded-batch test (1.00x). Forward alone: 4.6-10.9 ms/sample.
- one-CP-delayed async emission (pinned D2H + event, bitwise-exact max|d|=0.0): 0.90x — SLOWER.
  Root cause: sync no-writer path is only 23.6 ms/CP; the writer pool already overlaps emission,
  true sync idle is ~2 ms/CP while pinned-buffer mgmt + extra CPU copy costs ~2.7 ms/CP.
Stage profile (1ABB030, m16, sync ticks): chan 5.6 / norm 0.2 / fwd 14.5 / d2h 1.2 ms/CP.
Conclusion: at b32 the conv forward IS the floor; no per-CP scheduling trick pays. Ship as-is.
