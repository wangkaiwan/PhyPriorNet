#!/bin/bash
# photon-CT FAST container = the zero-cutoff+compress champion image with the base32 DISTILLED STUDENT
# weights swapped in (the app auto-detects base width from the ckpt, no code change). Student = 95.87 vs
# champion 96.15 (−0.28, near-lossless) but 2.1x net / ~1.55x e2e -> runtime 103.6s -> ~67s -> targets 1st.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
R=/home/kaiwang/doserad2026_workdir/runs
CTX=$(mktemp -d)
# extract the student EMA into a clean flat dose-net state_dict (like the champion photon.pt format)
conda run -n doserad python -c "
import torch
s=torch.load('$R/distill_photonct_b32_from4018/state.pt',map_location='cpu')
ema=s.get('ema', s.get('model'))
torch.save({'ema':ema,'step':s.get('step')}, '$CTX/photon.pt')
print('extracted student ema, stem width', ema['stem.weight'].shape[0])
" 2>&1 | grep -vE "Warning|warn"
BASE=doserad-photon:zerocut-m24-compress
printf 'FROM %s\nCOPY photon.pt /opt/algorithm/container/photon/weights/photon.pt\n' "$BASE" > "$CTX/Dockerfile"
echo "[build] doserad-photon:fast-student FROM $BASE (base32 student)"
sg docker -c "docker build -q -t doserad-photon:fast-student $CTX"
echo "[verify] baked env:"; sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-photon:fast-student" | grep -iE "CUTOFF_ZERO|COMPRESS|MARGIN|MODALITY"
echo "[save] -> doserad-photon-ct_fast-student.tar.gz"
sg docker -c "docker save doserad-photon:fast-student" | gzip -1 > "$SUB/doserad-photon-ct_fast-student.tar.gz"
echo "[done] $(du -h "$SUB/doserad-photon-ct_fast-student.tar.gz" | cut -f1)"
md5sum "$SUB/doserad-photon-ct_fast-student.tar.gz" | sed 's#/data.*/##'
rm -rf "$CTX"
echo "PHOTON-CT FAST-STUDENT BUILT"
