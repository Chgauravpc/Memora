"""
Path anchoring for the benchmark harness.

Hard requirement: every artefact the benchmark reads or writes must live inside the
repository directory (on the target server, /home/kenton/projects/memora). That includes
things that default to $HOME:

  - HuggingFace / sentence-transformers model cache (defaults to ~/.cache/huggingface)
  - Docker volumes (named volumes default to /var/lib/docker/volumes)

Both are redirected here / in docker-compose.benchmark.yml. Import this module before
anything that might touch those caches.
"""

from __future__ import annotations

import os
from pathlib import Path

# benchmarks/paths.py -> benchmarks/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data / output tree, all repo-relative
DATA_DIR = REPO_ROOT / "data"
LOCOMO_DIR = DATA_DIR / "locomo"
LOCOMO_JSON = LOCOMO_DIR / "locomo10.json"

RESULTS_DIR = REPO_ROOT / "results" / "locomo"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RAW_DIR = RESULTS_DIR / "raw"
LOG_DIR = REPO_ROOT / "logs"

# Caches that would otherwise land in $HOME
CACHE_DIR = REPO_ROOT / ".cache"
HF_CACHE = CACHE_DIR / "huggingface"

# Docker bind-mount targets
REDIS_DATA = DATA_DIR / "redis"
QDRANT_DATA = DATA_DIR / "qdrant"

_ALL_DIRS = [
    DATA_DIR, LOCOMO_DIR, RESULTS_DIR, CHECKPOINT_DIR, RAW_DIR,
    LOG_DIR, CACHE_DIR, HF_CACHE, REDIS_DATA, QDRANT_DATA,
]


def redirect_caches_into_repo() -> None:
    """
    Point every model/tokenizer cache at .cache/ inside the repo.

    Must run before `sentence_transformers` / `transformers` / `huggingface_hub` are
    imported -- they read these variables at import time. Existing values are respected
    if they already point inside the repo, so an operator can override deliberately.
    """
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    hf = str(HF_CACHE)
    for var in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TORCH_HOME",
    ):
        current = os.environ.get(var)
        if current and Path(current).resolve().is_relative_to(REPO_ROOT):
            continue
        os.environ[var] = hf


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def assert_contained(path: Path, what: str = "path") -> Path:
    """Refuse to read/write outside the repo."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(
            f"{what} {resolved} is outside the repository root {REPO_ROOT}. "
            "The benchmark is required to stay self-contained."
        )
    return resolved


def describe() -> str:
    return "\n".join([
        f"repo root     : {REPO_ROOT}",
        f"dataset       : {LOCOMO_JSON}",
        f"results       : {RESULTS_DIR}",
        f"model cache   : {HF_CACHE}",
        f"redis volume  : {REDIS_DATA}",
        f"qdrant volume : {QDRANT_DATA}",
    ])
