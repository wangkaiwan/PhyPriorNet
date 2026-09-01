#!/bin/bash
# Clean double-Gaussian proton-MRI E2E (2026-08-10): new whole-image 1x1x3 sCT (ref_1x1x3_samefield_whole2)
# + double-Gaussian proton-CT dose net (all75_r2_ft_double) jointly finetuned with DOSERAD_LATERAL_DOUBLE=1
# so the on-the-fly skin-entry PB prior is the Hong double-Gaussian (matches how the dose net was trained).
# SPEED: config sets vol_cache_n=80 -> caches all 75 patients' CPU arrays -> ~5x faster (2.7s->0.53s/step
# after a ~75-step warmup; 40k ~= 6h vs ~33h). GPU0. --resume if state.pt exists (safe re-launch).
# LOGGING: tee RAW stdout to train.log (NO tail buffering) so the live step is always readable.
set -u
cd /home/kaiwang/project/DoseRad2026
RUN=/home/kaiwang/doserad2026_workdir/runs/e2e_1x1x3_whole_protonmri_double
mkdir -p "$RUN"
CR="conda run -n doserad --no-capture-output python -u"
RES=""; [ -f "$RUN/state.pt" ] && RES="--resume" && echo "[double-mri] state.pt found -> --resume"
echo "[double-mri] launching on GPU0 (DOSERAD_LATERAL_DOUBLE=1, vol_cache_n=80), log -> $RUN/train.log"
DOSERAD_LATERAL_DOUBLE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 $CR scripts/train_dose_proton_e2e.py \
  --config configs/experiments/all75/e2e_1x1x3_whole_protonmri_double.yaml $RES 2>&1 \
  | grep --line-buffered -vE "win_data|FutureWarning|warn\(|cudart|GradScaler|scaler =" \
  | tee -a "$RUN/train.log"
echo "[double-mri] DONE"
