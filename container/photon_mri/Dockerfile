# DoseRAD2026 PHOTON-MRI dose-calculation algorithm — Grand-Challenge invoke container.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

LABEL org.grand-challenge.api-method="invoke"

WORKDIR /opt/algorithm
# GC requires a non-root user (validation rejects root containers)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# base image = Debian system python (torch preinstalled there); PEP 668 blocks pip without this
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    TORCHINDUCTOR_CACHE_DIR=/opt/algorithm/inductor_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DOSERAD_PHOTON_MARGIN=16 \
    DOSERAD_COMPRESS_OUTPUT=0 \
    DOSERAD_CUTOFF_ZERO=1 \
    DOSERAD_CONFIG=/opt/algorithm/container/photon_mri/weights/model_config.yaml \
    DOSERAD_WEIGHTS=/opt/algorithm/container/photon_mri/weights/photon_mri.pt \
    DOSERAD_CLF=/opt/algorithm/container/photon_mri/weights/clf_whole.pt \
    DOSERAD_MACHINE=/opt/algorithm/container/photon_mri/weights/beam_parameters.json

# --- python deps (pinned, no network at runtime; torch ships with the base image) ---
COPY pyproject.toml container/requirements.txt container/requirements-mri.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-mri.txt

# code: doserad package + acceleration + scripts (E2E model + sCT classifier live here)
COPY doserad/ ./doserad/
COPY accel/ ./accel/
COPY scripts/ ./scripts/
COPY data/proton_machine_generic.npz ./data/proton_machine_generic.npz
RUN pip install --no-cache-dir -e . \
    && mkdir -p /opt/algorithm/inductor_cache \
    && chown -R user:user /opt/algorithm/inductor_cache

# --- container layer LAST (brings weights/ once; code layers above cache independently) ---
COPY container/ ./container/

USER user
EXPOSE 4743
ENTRYPOINT ["python", "-m", "container.photon_mri.app"]
