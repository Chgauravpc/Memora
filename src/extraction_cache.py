"""
Content-addressed cache for Stage 3 LLM extraction.

WHY THIS EXISTS

Re-ingesting the same conversation twice, with no code change between runs, produced 769
memories one time and 431 the next -- same 419 turns, same 80.4% escalation rate, same
~19% empty-extraction rate. Only the store composition moved. `STAGE_3_TEMPERATURE` is 0.0,
which made every re-ingest reproducible under the previous non-reasoning model, but a
reasoning model spends hidden chain-of-thought before its visible reply and that trace is
not guaranteed bit-exact run to run.

The consequence is not a small one. Every A/B between two retrieval settings silently
compares two different memory stores, and the difference between stores is larger than any
retrieval effect being measured. The recorded score trajectory -- 64% then 32% then 48% --
contains at least one swing already attributed to exactly this.

Averaging over repeated ingests would cost N times the compute and still leave the noise in
the measurement. Caching removes it at the source: the first ingest fixes the store, and
every subsequent ingest of the same turns through the same prompt reproduces it exactly.

WHAT IS AND IS NOT IN THE KEY

The key is a hash of the fully-assembled prompt text plus provider, model, temperature and
max-tokens. Hashing the finished prompt rather than its ingredients is deliberate: the
prompt already contains the message, the conversation context, the speaker, the event date
and any Stage 2 hint, so a change to ANY of them -- including an edit to the prompt template
or the VALUE QUALITY block -- changes the key automatically. There is no list of inputs to
keep in sync, which is the usual way a cache like this goes wrong.

Two things the prompt text cannot see, carried separately:
  - `turn_number`, because it determines the generated `memory_id` but is not always in the
    prompt.
  - `EXTRACTION_CACHE_VERSION`, bumped by hand when the parsing/validation output shape
    changes.

WHAT IS NEVER CACHED

Only clean results. An empty list returned because the JSON was truncated, the schema was
wrong, or the API key was dead must never be frozen and replayed forever as "this turn had
nothing worth remembering" -- that is the blanket-except failure mode with a longer memory.
The caller passes the reason; see `EMPTY_REASON_CLEAN` in llm_extractor.

CONCURRENCY

The benchmark runs one OS subprocess per conversation, so several processes share this
directory. Each entry is its own file, written to a temporary name and moved into place with
os.replace (atomic on both Windows and POSIX), so a reader never observes a partial write
and two writers racing on the same key are harmless -- they are writing identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_key(
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    turn_number: int,
    version: str,
) -> str:
    """Stable hex digest identifying one extraction call.

    Field separators are newlines around a marker that cannot occur in the payloads, so two
    different field splits cannot collide into the same digest.
    """
    parts = [
        f"v={version}",
        f"provider={provider}",
        f"model={model}",
        f"temperature={temperature!r}",
        f"max_tokens={max_tokens}",
        f"turn={turn_number}",
        "prompt=",
        prompt,
    ]
    blob = "\n\x00--\x00\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


class ExtractionCache:
    """Disk-backed store of Stage 3 results, sharded by the first two hex characters.

    Sharding keeps any one directory to a few hundred entries, which matters on Windows
    where very large directories degrade noticeably.
    """

    def __init__(self, directory: str, enabled: bool = True):
        self.enabled = bool(enabled)
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0

        if self.enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                # A cache is an optimisation. Losing it must never stop an ingest.
                logger.warning("Extraction cache disabled -- cannot create %s: %s",
                               self.directory, exc)
                self.enabled = False

    def _path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[List[Dict]]:
        """Cached memories for `key`, or None for a miss.

        A corrupt or unreadable entry counts as a miss rather than an error, so a damaged
        cache degrades to a slow run instead of a failed one.
        """
        if not self.enabled:
            return None
        path = self._path_for(key)
        try:
            if not path.exists():
                self.misses += 1
                return None
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            memories = payload.get("memories")
            if not isinstance(memories, list):
                self.misses += 1
                return None
            self.hits += 1
            # Copy: callers mutate returned memories (adding speaker, user_id, event_date),
            # and a shared dict would leak those edits into whatever reads the cache next
            # within the same process.
            return [dict(m) for m in memories]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Extraction cache read failed for %s: %s", key[:12], exc)
            self.misses += 1
            self.errors += 1
            return None

    def put(self, key: str, memories: List[Dict], meta: Optional[Dict] = None) -> bool:
        """Store a clean result. Returns whether it was written."""
        if not self.enabled:
            return False
        path = self._path_for(key)
        payload = {"key": key, "memories": memories, "meta": meta or {}}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a concurrent reader sees either the old entry or the new
            # one, never a half-written file.
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp_name, path)
            except Exception:
                # Best effort: never leave temp files behind on a failed write.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            self.writes += 1
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Extraction cache write failed for %s: %s", key[:12], exc)
            self.errors += 1
            return False

    def stats(self) -> Dict[str, int]:
        """Counters for the results file.

        Surfaced deliberately: a cached run and a live run must be distinguishable after the
        fact, or the cache becomes another undisclosed thing a number depends on.
        """
        lookups = self.hits + self.misses
        return {
            "enabled": int(self.enabled),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate": round(self.hits / lookups, 4) if lookups else 0.0,
        }

    def clear(self) -> int:
        """Delete every entry. Returns how many files were removed."""
        if not self.directory.exists():
            return 0
        removed = 0
        for path in self.directory.glob("*/*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


_cache: Optional[ExtractionCache] = None


def get_extraction_cache() -> ExtractionCache:
    """Process-wide singleton, built from config on first use."""
    global _cache
    if _cache is None:
        from src.config import EXTRACTION_CACHE_DIR, EXTRACTION_CACHE_ENABLED
        _cache = ExtractionCache(EXTRACTION_CACHE_DIR, enabled=EXTRACTION_CACHE_ENABLED)
        if _cache.enabled:
            logger.info("Extraction cache active at %s", _cache.directory)
    return _cache


def reset_extraction_cache() -> None:
    """Drop the singleton so the next call rereads config. For tests."""
    global _cache
    _cache = None
