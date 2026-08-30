# DoseRAD2026 photon dose-calculation algorithm — Grand-Challenge invoke container.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

# GC invoke interface must be advertised or the platform won't route /invoke.
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
    DOSERAD_PHOTON_MARGIN=8 \
    DOSERAD_COMPRESS_OUTPUT=0 \
    DOSERAD_MODALITY=ct \
    DOSERAD_WEIGHTS=/opt/algorithm/container/photon/weights/photon.pt \
    DOSERAD_MACHINE=/opt/algorithm/container/photon/weights/beam_parameters.json

# --- python deps (pinned, no network at runtime; torch ships with the base image) ---
COPY pyproject.toml container/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- code: the doserad package + the acceleration layers ---
COPY doserad/ ./doserad/
COPY accel/ ./accel/
# proton machine kernel table (ProtonMachineData default _DEFAULT_NPZ = <root>/data/...); 1 MB, gitignored
COPY data/proton_machine_generic.npz ./data/proton_machine_generic.npz
RUN pip install --no-cache-dir -e . \
    && mkdir -p /opt/algorithm/inductor_cache \
    && chown -R user:user /opt/algorithm/inductor_cache

# --- container layer LAST (brings weights/ once; code layers above cache independently) ---
COPY container/ ./container/

USER user
EXPOSE 4743
ENTRYPOINT ["python", "-m", "container.photon.app"]
