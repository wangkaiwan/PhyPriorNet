#!/bin/bash
# base32 distilled student (from 4018f597) vs base48 champion (4018f597) on held16, SAME harness
# (cv_eval_photonct_full: live deploy channels + true full-grid GT + cutoff, margin 24). Is the student
# gamma-lossless? If yes -> build the fast container (photon-CT runtime -> 1st). GPU0.
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
F='grep -vE "win_data|Warning|warn|FutureWarning|cudart|Not enough|out\[idx"'
H=1ABB006,1ABB030,1ABB036,1ABB041,1ABB109,1ABB110,1ABB149,1ABB161,1THB002,1THB016,1THB027,1THB029,1THB048,1THB121,1THB191,1THB202
R=/home/kaiwang/doserad2026_workdir/runs
echo "########## CHAMPION base48 (4018f597) ##########"
CUDA_VISIBLE_DEVICES=0 $CR scripts/cv_eval_photonct_full.py --config configs/experiments/all75/_champ_b48_eval.yaml \
  --ckpt $R/docker_extracted/photon_ct_docker.pt --out $R/champ_b48_held16.csv --patients $H --margin 24 2>&1 | eval $F | grep -iE "mean|ALL|gamma|g1|>>>" | tail -6
echo "########## STUDENT base32 (distilled from 4018f597) ##########"
CUDA_VISIBLE_DEVICES=0 $CR scripts/cv_eval_photonct_full.py --config configs/experiments/all75/distill_photonct_b32_from4018.yaml \
  --ckpt $R/distill_photonct_b32_from4018/state.pt --out $R/student_b32_held16.csv --patients $H --margin 24 2>&1 | eval $F | grep -iE "mean|ALL|gamma|g1|>>>" | tail -6
echo "DISTILL-VS-CHAMP DONE"
