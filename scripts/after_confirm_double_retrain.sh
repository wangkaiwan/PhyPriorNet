#!/bin/bash
# When the N=16 double-Gaussian confirm finishes (frees GPU1), regenerate the proton PB prior cache with
# the DOUBLE-Gaussian (nuclear halo) lateral, then finetune the proton-CT dose net on it from the champion
# all75_r2_ft. Prior-only validation showed +3.5 abd / -0.9 lung (net +1.3 closer to GT). The residual net
# retrains on the cleaner prior. Eval (held16, WITH double prior) done separately. Autonomous.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough|out\[idx"'
DBLPRIOR=/data/kwang/doserad_cache_hdd/proton_prior_skinentry_double
DGLOG=/home/kaiwang/doserad2026_workdir/dg_confirm.log

echo "[dbl] waiting for the N=16 confirm to free GPU1 ..."
until grep -q ">>> PRIOR" "$DGLOG" 2>/dev/null || [ -z "$(pgrep -f 'diag_double_gaussian_prior.py 16')" ]; do sleep 120; done
sleep 20
echo "[dbl] ===== STEP 1: regen proton PB prior cache with DOUBLE-Gaussian -> $DBLPRIOR ====="
mkdir -p "$DBLPRIOR"
DOSERAD_LATERAL_DOUBLE=1 CUDA_VISIBLE_DEVICES=1 $CR scripts/build_proton_prior_skinentry.py \
  --out "$DBLPRIOR" --shard 0/1 2>&1 | eval $F | grep -E "\[|/|done|Error|Traceback" | tail -15
NP=$(find "$DBLPRIOR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
echo "[dbl] double prior cache: $NP patients"
[ "$NP" -ge 60 ] || { echo "[dbl] ABORT: prior cache incomplete ($NP)"; exit 1; }

echo "[dbl] ===== STEP 2: finetune proton-CT dose net on the double prior (init all75_r2_ft, 40k) ====="
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_proton.py \
  --config configs/experiments/all75/all75_r2_ft_double.yaml 2>&1 \
  | eval $F | grep -E "data|init|step|loss|val|best|saved|done|Error|Traceback" | tail -50
echo "[dbl] DOUBLE-GAUSSIAN RETRAIN DONE -> runs/all75_r2_ft_double/  (eval held16 WITH DOSERAD_LATERAL_DOUBLE=1 next)"
