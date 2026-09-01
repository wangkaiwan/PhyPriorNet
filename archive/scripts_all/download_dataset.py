"""Resumable HF snapshot download for DoseRAD2026 dataset.

Logs progress to stdout (redirected to logs/download.log when run in background).
Resumes on interruption because huggingface_hub uses cache + symlinks; rerunning
the script with the same local_dir continues where it left off.
"""

import logging
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/kwang/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data/kwang/huggingface/hub")
os.environ.setdefault("HF_XET_CACHE", "/data/kwang/huggingface/xet")

from huggingface_hub import snapshot_download

LOCAL_DIR = os.environ.get("DOSERAD_RAW", "/data/kwang/DoseRad2026_raw")
REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"

# Phase 1 = photon only (saves ~309 GB). Set DOSERAD_PATTERNS="*" for everything,
# or e.g. "proton/**" to add proton later. Default: photon + README.
_patterns_env = os.environ.get("DOSERAD_PATTERNS", "photon/**,README.md,*.json")
ALLOW_PATTERNS = None if _patterns_env.strip() == "*" else [
    p.strip() for p in _patterns_env.split(",") if p.strip()
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("doserad-dl")

os.makedirs(LOCAL_DIR, exist_ok=True)
log.info("Starting snapshot_download repo=%s -> %s", REPO_ID, LOCAL_DIR)

start = time.time()
log.info("allow_patterns=%s", ALLOW_PATTERNS)
path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=ALLOW_PATTERNS,
    max_workers=8,
    tqdm_class=None,
)
elapsed = time.time() - start
log.info("Done in %.1f min, local path: %s", elapsed / 60, path)
