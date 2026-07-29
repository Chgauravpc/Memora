"""
Preflight: verify the box can actually run the benchmark before burning hours on it.

Checks in dependency order and keeps going after failures so one run surfaces every
problem. Exit code 0 only when nothing is broken.

The Qdrant check matters more than it looks: if the vector store is unreachable, Memora
degrades SILENTLY to Phase 1 retrieval (src/__init__.py and memory_system.py swallow the
ImportError/connection error and log a warning). The benchmark would complete and produce
plausible-looking but meaningless numbers. Fail loudly here instead.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .paths import (HF_CACHE, LOCOMO_JSON, REPO_ROOT, describe,
                    ensure_dirs, redirect_caches_into_repo)

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
_failures: list[str] = []
_warnings: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    if status == BAD:
        _failures.append(label)
    elif status == WARN:
        _warnings.append(label)


def check_python() -> None:
    v = sys.version_info
    # pathlib.Path.is_relative_to (used for path containment) is 3.9+.
    if v >= (3, 9):
        report(OK, "python version", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report(BAD, "python version", f"{v.major}.{v.minor} - need >= 3.9")


def check_paths() -> None:
    ensure_dirs()
    redirect_caches_into_repo()
    print(describe())
    if os.getenv("HF_HOME", "").startswith(str(HF_CACHE)):
        report(OK, "model cache inside repo", os.environ["HF_HOME"])
    else:
        report(WARN, "model cache", f"HF_HOME={os.getenv('HF_HOME')}")

    expected = "/home/kenton/projects/memora"
    actual = str(REPO_ROOT).replace("\\", "/")
    if actual == expected:
        report(OK, "repo location", actual)
    else:
        report(WARN, "repo location", f"{actual} (target server path is {expected})")

    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    if free_gb >= 10:
        report(OK, "free disk", f"{free_gb:.0f} GB")
    else:
        report(WARN, "free disk", f"{free_gb:.1f} GB - model + volumes want ~10 GB")


def check_deps() -> None:
    required = {
        "redis": "redis",
        "qdrant_client": "qdrant-client",
        "sentence_transformers": "sentence-transformers",
        "numpy": "numpy",
        "dotenv": "python-dotenv",
    }
    for module, pkg in required.items():
        try:
            __import__(module)
            report(OK, f"import {module}")
        except ImportError as exc:
            report(BAD, f"import {module}", f"pip install {pkg} ({exc})")


def check_redis() -> None:
    try:
        import redis as redis_lib
        from src.config import REDIS_HOST, REDIS_PORT
    except ImportError as exc:
        report(BAD, "redis client", str(exc))
        return

    try:
        client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=5)
        client.ping()
        report(OK, "redis reachable", f"{REDIS_HOST}:{REDIS_PORT}")

        databases = int(client.config_get("databases").get("databases", 16))
        workers = int(os.getenv("BENCH_WORKERS", "4"))
        if databases >= workers:
            report(OK, "redis logical DBs", f"{databases} >= {workers} workers")
        else:
            report(BAD, "redis logical DBs",
                   f"{databases} DBs but {workers} workers requested - workers would "
                   f"share a DB and cross-contaminate. Raise `databases` in redis.conf.")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "redis reachable", f"{exc} - run: docker compose -f "
                                       f"docker-compose.benchmark.yml up -d")


def check_qdrant() -> None:
    try:
        from qdrant_client import QdrantClient
        from src.config import QDRANT_HOST, QDRANT_PORT
    except ImportError as exc:
        report(BAD, "qdrant client", str(exc))
        return

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        client.get_collections()
        report(OK, "qdrant reachable", f"{QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "qdrant reachable",
               f"{exc} - WITHOUT Qdrant Memora silently degrades to Phase 1 retrieval "
               f"and the benchmark numbers become meaningless")


def check_embeddings() -> None:
    try:
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL
    except ImportError as exc:
        report(BAD, "embedding model", str(exc))
        return
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(HF_CACHE))
        dim = model.get_sentence_embedding_dimension()
        from src.config import EMBEDDING_DIMENSION
        if dim == EMBEDDING_DIMENSION:
            report(OK, "embedding model", f"{EMBEDDING_MODEL} ({dim}d)")
        else:
            report(BAD, "embedding model", f"dim {dim} != config {EMBEDDING_DIMENSION}")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "embedding model", f"{exc} (first run needs internet to download)")


def check_llm() -> None:
    try:
        from .llm import LLMClient
        client = LLMClient()
    except Exception as exc:  # noqa: BLE001
        report(BAD, "LLM client", str(exc))
        return

    reply = client.chat(user="Reply with the single word: ready", max_tokens=8)
    if reply and "ready" in reply.lower():
        report(OK, "LLM live call", f"{client.provider}/{client.model}")
    elif reply:
        report(WARN, "LLM live call", f"unexpected reply: {reply[:60]!r}")
    else:
        report(BAD, "LLM live call", "all attempts failed - check key and rate limits")


def check_stage3() -> None:
    try:
        from src.config import STAGE_3_ENABLED, LLM_PROVIDER, LLM_EXTRACTION_MODEL
    except ImportError as exc:
        report(BAD, "stage 3 config", str(exc))
        return
    if STAGE_3_ENABLED:
        report(OK, "stage 3 extraction", f"{LLM_PROVIDER}/{LLM_EXTRACTION_MODEL}")
    else:
        report(WARN, "stage 3 extraction",
               "DISABLED - extraction will be regex-only, which on out-of-domain LoCoMo "
               "text will extract very little")


def check_dataset() -> None:
    if not LOCOMO_JSON.exists():
        report(BAD, "dataset", f"missing {LOCOMO_JSON} - run: python -m benchmarks.download")
        return
    try:
        from .dataset import iter_answerable, load_conversations
        convs = load_conversations()
        turns = sum(len(c.turns) for c in convs)
        qs = sum(len(list(iter_answerable(c))) for c in convs)
        report(OK, "dataset", f"{len(convs)} conversations, {turns} turns, {qs} gradable Qs")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "dataset", str(exc))


def main() -> int:
    # src imports read .env; make sure the repo is importable when run from anywhere.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    print("=" * 66)
    print("Memora LoCoMo benchmark preflight")
    print("=" * 66)
    check_python()
    check_paths()
    print()
    check_deps()
    print()
    check_redis()
    check_qdrant()
    check_embeddings()
    print()
    check_stage3()
    check_llm()
    print()
    check_dataset()

    print()
    print("=" * 66)
    if _failures:
        print(f"{len(_failures)} FAILURE(S): {', '.join(_failures)}")
        print("Fix these before running the benchmark.")
        return 1
    if _warnings:
        print(f"ready, with {len(_warnings)} warning(s): {', '.join(_warnings)}")
        return 0
    print("all checks passed - ready to run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
