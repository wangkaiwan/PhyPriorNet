#!/bin/bash
# Double-Gaussian proton-CT docker = the 1st-place all75r2-compressed image (quantise + compress, the proton
# defaults we KEEP — proton is write-bound so compression helps, and zero-cutoff was WORSE for proton) with:
#  - proton.pt <- all75_r2_ft_double (the halo-prior finetune, 40k, converged)
#  - ENV DOSERAD_LATERAL_DOUBLE=1 (build_ray uses the double prior at inference == training)
# Held16 in-sample is ~flat (+0.05 abd) because the halo is cropped out + net absorbs it; upload is a
# HELD-OUT overfitting probe (user's idea): the physically-correct prior may generalize better on the test.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
R=/home/kaiwang/doserad2026_workdir/runs
CTX=$(mktemp -d)
conda run -n doserad python -c "
import torch; s=torch.load('$R/all75_r2_ft_double/best.pt',map_location='cpu')
torch.save({'ema':s.get('ema',s.get('model')),'step':s.get('step')}, '$CTX/proton.pt')  # keep 'ema' key: app does sd.get('ema')
print('  staged double proton.pt with ema key, step', s.get('step'))" 2>&1 | grep -vE "Warning|warn"
printf 'FROM doserad-proton:all75r2-compressed\nCOPY proton.pt /opt/algorithm/container/proton/weights/proton.pt\nENV DOSERAD_LATERAL_DOUBLE=1\n' > "$CTX/Dockerfile"
echo "[build] doserad-proton:double FROM all75r2-compressed + double model + LATERAL_DOUBLE=1"
sg docker -c "docker build -q -f $CTX/Dockerfile -t doserad-proton:double $CTX"
echo "[verify] env (expect LATERAL_DOUBLE=1, NO CUTOFF_ZERO, modality ct):"
sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-proton:double" | grep -iE "LATERAL_DOUBLE|CUTOFF|COMPRESS|MODALITY"
echo "[save] -> doserad-proton-ct_double.tar.gz"
sg docker -c "docker save doserad-proton:double" | gzip -1 > "$SUB/doserad-proton-ct_double.tar.gz"
echo "[done] $(du -h "$SUB/doserad-proton-ct_double.tar.gz"|cut -f1)"; md5sum "$SUB/doserad-proton-ct_double.tar.gz" | sed 's#/data.*/##'
rm -rf "$CTX"
echo "DOUBLE PROTON-CT DOCKER BUILT"
