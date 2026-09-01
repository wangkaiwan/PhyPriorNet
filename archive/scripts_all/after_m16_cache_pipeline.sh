#!/bin/bash
# margin-16 pipeline (GPU1): wait for the m16 cache, then ② photon-CT ft+eval → ③ photon-MRI PURE ft+eval
# → ④ multi-modal (ct_mix 0.5) + dual eval. Each eval = held16 official gamma @ margin-16.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/all75
R=/home/kaiwang/doserad2026_workdir/runs
CACHE_LOG=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/brh3dlcll.output
M16=/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m16
CLFMRAUG=/data/kwang/sct_classify_runs/clf_whole_mraug/best.pt
F='grep -vE "win_data|out\[idx|Warning|warn|FutureWarning|cudart|GradScaler|scaler =|Not enough"'
ev_ct(){ DOSERAD_PHOTON_MARGIN=16 DOSERAD_W_OVERRIDE="$1" CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_official_held16.py photon_ct 16 2>&1 | eval $F | grep -E ">>>"; }
ev_mri(){ DOSERAD_PHOTON_MARGIN=16 DOSERAD_W_OVERRIDE="$1" DOSERAD_CLF_OVERRIDE=$CLFMRAUG CUDA_VISIBLE_DEVICES=1 $CR scripts/eval_official_held16.py photon_mri 16 2>&1 | eval $F | grep -E ">>>"; }

echo "[m16] rebuilding crop cache -> HDD (dose targets already exist; disk-full retry) ..."
CUDA_VISIBLE_DEVICES=1 $CR scripts/build_photon_cache_m24.py --margin 16 --shard 0/1 \
  --src /data/kwang/doserad_cache/photon_dose_m16 --out "$M16" 2>&1 | eval $F | grep -E "done|/|Error|Traceback" | tail -12
[ "$(find "$M16" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)" -ge 60 ] || { echo "[m16] ABORT crop incomplete ($(find "$M16" -maxdepth 1 -mindepth 1 -type d 2>/dev/null|wc -l))"; exit 1; }
echo "[m16] crop cache ready ($(find "$M16" -maxdepth 1 -mindepth 1 -type d|wc -l) pt)"

echo "############### ② photon-CT margin-16 finetune ###############"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train.py --config $CFG/ftm16_from_best.yaml 2>&1 | eval $F | grep -E "init|step (1|[0-9]000)/|val|best|done" | tail -6
echo "######## ② CT @ m16 (vs m24 98.78 / m8 95.27) ########"; ev_ct $R/ftm16_from_best/snap_020000.pt

echo "############### ③ photon-MRI margin-16 PURE finetune (ct_mix 0) ###############"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_e2e.py --config $CFG/ftm16_photonmri.yaml 2>&1 | eval $F | grep -E "init|ct-mix|step (1|[0-9]000)/|val|best|done" | tail -6
echo "######## ③ MRI @ m16 (vs m24 96.35 / m8 91.65) ########"; ev_mri $R/ftm16_photonmri/state.pt

echo "############### ④ multi-modal (ct_mix 0.5) ###############"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 $CR scripts/train_dose_e2e.py --config $CFG/mm_ftm16_photonmri.yaml 2>&1 | eval $F | grep -E "init|ct-mix|step (1|[0-9]000)/|val|best|done" | tail -6
$CR -c "import torch; d=torch.load('$R/mm_ftm16_photonmri/state.pt',map_location='cpu'); e=d['ema']; torch.save({'ema':{k[5:]:v for k,v in e.items() if k.startswith('dose.')},'step':d.get('step')},'$R/mm_ftm16_photonmri/dose_ct_extracted.pt')" 2>&1 | grep -v Warning
echo "######## ④ MRI @ m16 (multimodal) ########"; ev_mri $R/mm_ftm16_photonmri/state.pt
echo "######## ④ CT @ m16 (extracted, vs dedicated ② ) ########"; ev_ct $R/mm_ftm16_photonmri/dose_ct_extracted.pt
echo "M16 PIPELINE DONE"
