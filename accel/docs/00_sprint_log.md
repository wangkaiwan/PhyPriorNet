# Acceleration sprint log (GPU1 dedicated, from 2026-07-17)

## Profiles (RTX 5090, GPU-synced, 1ABB006)
- **Proton** 120 ms/beamlet: PB prior 44 (37%) + WEPL 41 (34%) + forward 24 (20%) + io 9 + stack 3.
  Physics build = 71%. Per-ray dedup rejected (sibling bbox IoU 0.52). → batch WEPL/PB engines.
- **Photon** 250 ms/CP: forward(b<=8) 194 (78%) + photon_channels 56 (22%) + normalize 0.5.
  Forward-dominated because effective batch ~1.2 (padded-voxel cap) → masked-GN batching is the win.

## IDD step 3 (signed range) — VERDICT: no scalar-calibration fix
Proton-MRI signed dR80 = **-0.08 mm** (mean), 8 patients +, 8 -; |dR80| 2.35 mm.
Error is UNBIASED / symmetric scatter, not a systematic overshoot. A global density-scale
calibration would NOT reduce it. Documented negative check. (abdomen +0.37, lung -0.54 mm.)

## masked-GN v2 — padded-batch == per-sample (equivalence)
accel/masked_gn.py: (1) GroupNorm stats over each sample's 16-aligned valid box (matches the
deployed per-sample pad semantics — the KEY fix; a "pure valid region" v1 gave 17% error because
deployment itself pads to /16), (2) re-zero padding after every leaf module (hook) to stop conv
edge contamination. Test accel/test_masked_gn.py (4 different-size CP crops):
  fp32 max rel diff 1.98e-4 (float-assoc residual), **amp (deploy path) 9.3e-4 < 2e-3 PASS**.
→ Unlocks the 1.4-2.2x compile/batching that the S12 study found broke the 0.1% bar.

## Batching investigation — RESOLVED as a dead end for photon (2026-07-17)
- masked-GN padded batch: 0.98x (accuracy PASS 0.006%). Diagnostic: SAME-size crop batch4 = 1.7x
  -> not compute-bound per se, but real plans have heterogeneous + LARGE crops.
- Size-bucketed plain batching (padded-16 buckets, 27-61/patient): still 1.00x. Root cause: photon
  CP crops are ~6M voxels each; a memory-safe batch of them is ~1 on 24-31GB -> the big buckets
  (which hold most CPs) run batch-1 anyway. **Photon-forward batching is MEMORY-bound, dead end.**
- => photon-forward levers are: (a) torch.compile kernel fusion (single shape, safe), (b) a SMALLER
  network via distillation (fewer FLOPs on the same big volume = direct win + then batchable).
- Proton is different: build-bound (71%); batch the WEPL/PB engines. masked-GN stays useful there
  (proton plain-batch broke 0.1% in S12) if we ever batch the proton forward (only 20%).

## ⭐ torch.compile — THE photon-forward win (2026-07-17)
- eager 50.7ms -> compile(max-autotune) 16.8ms = **3.0x** on one shape.
- **dynamic=True: 3.05x across 4 DIFFERENT shapes** (156.7->51.4ms) with only shape-0 warmed up —
  NO per-shape recompile storm. Solves the S12 compile-storm blocker.
- Deployment recipe: torch.compile(net, dynamic=True), warm up on 1 dummy shape during the FREE
  /health model-load phase, TORCHINDUCTOR_CACHE_DIR persisted -> cross-run kernel cache.
- Accuracy: plan-level compiled-vs-eager rel-diff = 0.0149% (PASS <0.1%). CONFIRMED.
- Impact: photon forward 194->~64ms/CP; + channels 56ms -> ~120ms/CP @5090 -> ~0.3-0.42s/beam A10G,
  inside the 1s gate with margin. Makes distillation a BONUS (compound) rather than a necessity.

## compile accuracy — verified on the SCORED metric (plan gamma 1%/1mm), 2026-07-17
Full fold-0 val (16 patients), eager vs torch.compile(dynamic=True), gamma_compile_check.py:
  MEAN plan gamma1/1: eager 94.01 -> compiled 94.02 (delta +0.009); max per-patient |delta| 0.08.
=> compile is dosimetrically LOSSLESS on the scored metric (not just dose rel-diff). Confirmed win:
   3.05x forward at ~0 gamma cost. Autocast(AMP) was already deployed; compile is NEW (S12 rejected
   compile+BATCHING for 0.74% via GroupNorm-padding — different config; single-crop compile is clean).

## Proton compile probe + gate status (2026-07-17 eve)
- proton NET compile: 15.1 -> 5.7 ms = 2.65x (same as photon; gamma-verify pending).
- WEPL 37.7ms / PB 38.6ms eager: NOT directly compilable (take image-object + numpy args) ->
  build-side speedup needs batching-refactor (concat K beamlets' voxels into one grid_sample/kernel).
- **GATE STATUS: BOTH tasks pass the 1 s/beam gate with compile alone** (A10G x2.5-3.5 scale):
  photon ~0.3-0.4 s/beam, proton ~0.28-0.4 s/beam. The exclusion risk is removed.
- Build-side proton batching = a RANKING optimization (runtime 2/7 + 1st tiebreaker), not gate
  compliance. Higher effort/risk. RECOMMENDATION: defer until prelim measures the real organizer
  stopwatch + our real rank, then decide ROI.

## Proton build — per-RAY geometry sharing (2026-07-17, INVEST decision)
accel/proton_ray_batch.py: a ray's beamlets share src/SSD/WEPL/lateral (energy-independent); only
the idd/sigma kernel lookup differs. Compute geometry once on the ray's union bbox, slice per beamlet.
Test (30 rays/60 beamlets, 1ABB006): **2.01x exact** (40.6->20.2 ms/beamlet, rel-diff 5.7e-6 vs the
per-beamlet engine). 540 rays x 2 energies -> applies to the whole plan.
FURTHER (fold into container inference): the deploy path computes WEPL TWICE per beamlet — once for
the ch1 INPUT channel (_wepl_on_density) and again inside the PB engine (_wepl_crop). Same integral.
A ray-centric inference computes WEPL ONCE and feeds BOTH the channel (all siblings) and the PB dose
(all siblings) -> up to 4x on WEPL + 2x on PB. To be built into the container pipeline.

## Proton fast pipeline — CORRECTED (2026-07-17, honest)
BUG FOUND via full-channel + real-gamma check: the ch2 PB prior needs the SKIN-ENTRY engine
(skin-referenced WEPL + `entered` gate, eff_depths WITHOUT rad_depth_offset). My first build_ray
used the non-skinentry PB -> ch2 rel-diff 0.5 -> plan gamma 96.5 (vs 98.7). Also energy channel
missed /_E_SCALE(250) -> gamma 8.4. Both fixed; all 5 channels now <3e-4 vs deploy-exact, plan
gamma 98.7 = baseline (LOSSLESS). Also: ch1(source-ref) and PB(skin-ref) use DIFFERENT WEPLs, so
"WEPL once for both" was wrong (2 raymarches needed).
CLEAN A/B (both compiled net): per-beamlet 44.0 -> ray-centric 34.4 ms/bl = **1.28x** (not 3.24x).
Reason: ray-sibling UNION bbox has ~2x voxels (IoU 0.52), so per-ray WEPL FLOPs ~= 2 beamlets;
sharing only amortizes launch overhead. The real proton win is net compile (forward 15->6ms,
gamma-verified lossless via the 98.7 full-plan). Proton now ~34 ms/bl -> ~0.1-0.15 s/beam A10G.
NEXT (diminishing): fuse the 2 WEPL raymarches into 1 (skin-ref = post-skin part of source-ref)
-> ~2x on the WEPL portion.

## Proton WEPL fusion (2026-07-17) — recovered the sharing win
accel/wepl_fused.py: one grid_sample -> source-ref WEPL (ch1) + skin-ref WEPL + entered (PB), which
deploy computes as 2 raymarches. build_ray uses it. All 5 channels <2e-3 vs deploy; plan gamma
98.7 = baseline (LOSSLESS). A/B: ray-centric 34.4 -> 24.6 ms/bl; speedup 1.28x -> **1.78x**. Full
proton fast pipeline (compiled net + fused ray-centric): **24.6 ms/beamlet, ~0.06-0.09 s/beam A10G**.

## Container proton pipeline (2026-07-18 overnight) — DEPLOY-REAL (no GT)
- GC invoke contract layer container/proton/gc_invoke.py: 10-slot I/O, JoinSeries 4D, compression,
  cutoff-zero, placeholders, grid-match. Mock harness test_contract.py: 28/28 checks PASS.
- DEPLOY GOTCHA solved: no GT bbox. geom_bbox_proton (central-ray WEPL march -> true geometric
  range + lateral margin) gives a superset bbox; then PB-threshold TIGHT crop ({PB>0.01*max}+4,
  matches training GT crop) so DoseUNet3D GroupNorm sees the training crop distribution.
  Without tight-crop: gamma 70 (GroupNorm crop-size sensitivity). With: **gamma 98.3** (vs GT-bbox
  98.7 — the 98.7 is unachievable at deploy; 98.3 is the honest deploy number). container/proton/
  predict.py. Speed being tuned (geom-box WEPL cost).

## Proton container — END-TO-END VALIDATED (2026-07-18 ~02:30)
- test_contract.py: 28/28 GC-contract checks (10-slot 4D JoinSeries, compression, cutoff, grid,
  placeholders). test_predict.py: deploy gamma 98.3 (geom bbox + PB tight-crop, no GT, all beams).
  test_integration.py (20-beam realistic sub-batch): process_run 5.2s, all plumbing PASS (4D, frame
  count, source-grid, nonzero, sane dose range), NO OOM.
- Memory bug fixed: output frames are FULL-grid (161 MB @ 498x493x164); the 1080-beam single-stack
  OOM (174 GB) was a TEST artifact (organizers sub-batch). Lazy just-in-time frame materialization.
- Files: container/proton/{gc_invoke,predict,geom_bbox,app,test_contract,test_predict,test_integration,
  Dockerfile,build.sh}. Ready to `docker build` (build.sh stages gitignored weights + machine npz).
- Photon container = TODO (same skeleton; photon_channels + compiled net; forward-dominated).

## Proton deploy speed tuning (2026-07-18 ~06:00)
Deploy steady-state (warm inductor cache, run2/3): 91.5 ms/beamlet @ lateral 28mm. Component profile:
geom_bbox 0.2 + build(WEPL on geom box) 29 + tight+net ~63 ms; geom box 5.9x inflated vs tight crop.
Lateral sweep (gamma-verified, abdomen 1ABB006 / lung 1THB002):
  18mm -> 54.8 ms, gamma 98.0 (clips penumbra, -0.3) REJECT
  24mm -> 77.0 ms, gamma 98.3 (abd) / 97.3 (lung) = no clip
  28mm -> 91.5 ms, gamma 98.3 (abd) / 97.4 (lung)
Lung 24 vs 28 = 97.3 vs 97.4 (negligible) -> 24mm does NOT clip; the lung -0.6 deploy gap is the
inherent PB-tight-crop vs GT-crop cost (heterogeneous), not lateral. LOCK lateral=24mm: 1.2x free
(91.5->77 ms/beamlet), gamma unchanged. A10G ~0.19-0.27 s/beam. HONEST deploy: abd 98.3 / lung 97.3.
