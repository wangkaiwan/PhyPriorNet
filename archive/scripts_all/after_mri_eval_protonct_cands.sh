#!/bin/bash
# Wait for the photon-MRI margin-8 finetune to free GPU1, then eval the proton-CT candidates
# (multi-modal protonct_from_ft_mm, protonct_from_ft_mraug) vs the champion all75_r2_ft (97.50) on held16.
set -u
cd /home/kaiwang/project/DoseRad2026
FT=/tmp/claude-1001/-home-kaiwang-project-DoseRad2026/5566bf81-6ba9-48f3-baf6-e17189e3dfc9/tasks/b9gijrymm.output
echo "[cand] waiting for photon-MRI finetune to finish ..."
until grep -qE "done ->|20000/20000" "$FT" 2>/dev/null; do sleep 90; done
sleep 20
E="conda run -n doserad --no-capture-output python -u scripts/eval_deploy_held16.py proton_ct"
X=/home/kaiwang/doserad2026_workdir/runs/docker_extracted
for W in "all75_r2_ft(champion=97.50):$X/../all75_r2_ft/state.pt" \
         "protonct_from_ft_mm(multimodal):$X/protonct_from_ft_mm.pt" \
         "protonct_from_ft_mraug(mrift):$X/protonct_from_ft_mraug.pt"; do
  name=${W%%:*}; wt=${W#*:}
  echo "######## $name ########"
  env DOSERAD_W_OVERRIDE=$wt CUDA_VISIBLE_DEVICES=1 $E 2>&1 \
    | grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough" | grep -E "γ1/1 ALL|ALL|held16 baseline" | tail -3
done
echo "[cand] PROTONCT CANDIDATES DONE"
