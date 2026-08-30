#!/bin/bash
# photon-CT FAST container WITHOUT compression: compression doesn't help photon (compute-bound; write is
# hidden under compute) and was leaderboard-verified slightly SLOWER on photon-MRI. So build from the
# COMPRESS=0 zero-cutoff image + base32 distilled student. Runtime win = distillation (2.1x net), not zlib.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
R=/home/kaiwang/doserad2026_workdir/runs
CTX=$(mktemp -d)
conda run -n doserad python -c "
import torch
s=torch.load('$R/distill_photonct_b32_from4018/state.pt',map_location='cpu')
ema=s.get('ema', s.get('model'))
torch.save({'ema':ema,'step':s.get('step')}, '$CTX/photon.pt')
print('student ema extracted, base', ema['stem.weight'].shape[0])
" 2>&1 | grep -vE "Warning|warn"
printf 'FROM doserad-photon:zerocut-m24\nCOPY photon.pt /opt/algorithm/container/photon/weights/photon.pt\n' > "$CTX/Dockerfile"
echo "[build] doserad-photon:fast-student-nocompress FROM doserad-photon:zerocut-m24 (COMPRESS=0)"
sg docker -c "docker build -q -t doserad-photon:fast-student-nocompress $CTX"
echo "[verify] baked env (expect COMPRESS=0, CUTOFF_ZERO=1, MARGIN=24):"
sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-photon:fast-student-nocompress" | grep -iE "CUTOFF_ZERO|COMPRESS|MARGIN|MODALITY"
echo "[save] -> doserad-photon-ct_fast-student_nocompress.tar.gz"
sg docker -c "docker save doserad-photon:fast-student-nocompress" | gzip -1 > "$SUB/doserad-photon-ct_fast-student_nocompress.tar.gz"
echo "[done] $(du -h "$SUB/doserad-photon-ct_fast-student_nocompress.tar.gz" | cut -f1)"
md5sum "$SUB/doserad-photon-ct_fast-student_nocompress.tar.gz" | sed 's#/data.*/##'
rm -rf "$CTX"
echo "PHOTON-CT FAST-STUDENT NOCOMPRESS BUILT"
