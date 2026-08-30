#!/bin/bash
# margin-16 dockers, NO compression:
#  ① photon-CT  = zerocut-m24 photon image + ftm16_from_best base48 dose net + MARGIN=16 (held16 98.55)
#  ② photon-MRI = zerocut-m24 photon-mri image + mm_ftm16 FULL multimodal E2E (synth/sCT + dose) + MARGIN=16
#     (held16 96.46 — the multimodal MRI is the best margin-16 photon-MRI; clf_whole_mraug unchanged,
#     coarse recipe matches). Both keep CUTOFF_ZERO=1, COMPRESS=0 (compression doesn't help compute-bound
#     photon, verified slightly slower). Weights = each ckpt's EMA extracted clean.
set -eu
cd "$(git rev-parse --show-toplevel)"
SUB=/data/kwang/doserad_submissions
R=/home/kaiwang/doserad2026_workdir/runs
CTX=$(mktemp -d)

extract(){ conda run -n doserad python -c "
import torch; s=torch.load('$1',map_location='cpu'); torch.save({'ema':s.get('ema',s.get('model')),'step':s.get('step')},'$2')
sd=s.get('ema',s.get('model')); print('  extracted', '$2', 'nkeys', len(sd))" 2>&1 | grep -vE "Warning|warn"; }

# ① margin-16 photon-CT
extract $R/ftm16_from_best/snap_020000.pt "$CTX/photon.pt"
printf 'FROM doserad-photon:zerocut-m24\nCOPY photon.pt /opt/algorithm/container/photon/weights/photon.pt\nENV DOSERAD_PHOTON_MARGIN=16\n' > "$CTX/Dockerfile"
echo "[build] ① doserad-photon:m16-zerocut FROM zerocut-m24 + ftm16_from_best + MARGIN=16"
sg docker -c "docker build -q -f $CTX/Dockerfile -t doserad-photon:m16-zerocut $CTX"
sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-photon:m16-zerocut" | grep -iE "MARGIN|CUTOFF|COMPRESS"
sg docker -c "docker save doserad-photon:m16-zerocut" | gzip -1 > "$SUB/doserad-photon-ct_m16_zerocut_nocompress.tar.gz"
echo "[done ①] $(du -h "$SUB/doserad-photon-ct_m16_zerocut_nocompress.tar.gz"|cut -f1)"; md5sum "$SUB/doserad-photon-ct_m16_zerocut_nocompress.tar.gz" | sed 's#/data.*/##'

# ② margin-16 photon-MRI (multimodal, full E2E synth+dose)
extract $R/mm_ftm16_photonmri/state.pt "$CTX/photon_mri.pt"
printf 'FROM doserad-photon-mri:zerocut-m24\nCOPY photon_mri.pt /opt/algorithm/container/photon_mri/weights/photon_mri.pt\nENV DOSERAD_PHOTON_MARGIN=16\n' > "$CTX/Dockerfile2"
echo "[build] ② doserad-photon-mri:m16-mm-zerocut FROM zerocut-m24 + mm_ftm16 E2E + MARGIN=16"
sg docker -c "docker build -q -f $CTX/Dockerfile2 -t doserad-photon-mri:m16-mm-zerocut $CTX"
sg docker -c "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' doserad-photon-mri:m16-mm-zerocut" | grep -iE "MARGIN|CUTOFF|COMPRESS"
sg docker -c "docker save doserad-photon-mri:m16-mm-zerocut" | gzip -1 > "$SUB/doserad-photon-mri_m16_mm_zerocut_nocompress.tar.gz"
echo "[done ②] $(du -h "$SUB/doserad-photon-mri_m16_mm_zerocut_nocompress.tar.gz"|cut -f1)"; md5sum "$SUB/doserad-photon-mri_m16_mm_zerocut_nocompress.tar.gz" | sed 's#/data.*/##'
rm -rf "$CTX"
echo "MARGIN16 DOCKERS BUILT"
