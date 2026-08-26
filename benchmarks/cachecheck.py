"""
Is the Stage 3 extraction cache actually warm for this conversation?

WHY THIS EXISTS

The cache is what makes a re-ingest free and reproducible, and until now the only way to
find out whether it was working was to run an ingest and inspect the store size afterwards.
That is a bad way to learn it: a cache that silently misses looks exactly like a cache that
hit, right up until the store comes back at 77 memories instead of 751 -- by which point the
previous store is gone, because the benchmark Redis does not persist.

This replays the real turns through the real extractor with the network amputated, and
counts how many escalations would have been served from disk. No API calls, no Redis, no
Qdrant, no writes of any kind. It answers three different failures apart:

  escalations = 0            Stage 3 is not running at all. The LLM extractor failed to
                             construct (missing package, import error), so the cache is
                             never even consulted -- extraction falls back to Stage 1/2
                             regex, which on narrative dialogue yields very little.

  escalations > 0, hits = 0  The cache is disabled, pointed at the wrong directory, or the
                             keys no longer match because the prompt, model, temperature or
                             token cap changed since it was populated.

  hits ~= escalations        Warm. A re-ingest will cost approximately nothing and rebuild
                             the same store.

USAGE

    python -m benchmarks.cachecheck                      # first conversation in the dataset
    python -m benchmarks.cachecheck --sample-id conv-26
    python -m benchmarks.cachecheck --sample-id conv-26 --max-turns 50   # quick probe

Run it before `--ingest-only` or before a graded run, not after.
"""

from __future__ import annotations

# Cache redirection must precede any transformers import, as in worker.py.
from .paths import redirect_caches_into_repo  # noqa: E402
redirect_caches_into_repo()

import os  # noqa: E402

# Probe the cache the way a benchmark run would see it.
#
# EXTRACTION_CACHE_ENABLED defaults to False so that importing Memora as a library changes
# no behaviour; benchmarks/runner.py turns it on for its workers. Without the same default
# here, this tool answers a question nobody asked: with the cache disabled `cache.get` is
# never called, every escalation falls through to the API, and the report says 0% coverage
# no matter how warm the cache actually is -- indistinguishable from keys that genuinely do
# not match, which is the one thing this exists to tell apart.
#
# Set before any `src` import, because src/config.py freezes environment into module
# constants at import time. setdefault, not assignment, so an explicit
# EXTRACTION_CACHE_ENABLED=false on the command line still wins.
os.environ.setdefault("EXTRACTION_CACHE_ENABLED", "true")

import argparse  # noqa: E402
import logging  # noqa: E402
from typing import Optional  # noqa: E402

from .dataset import load_conversations  # noqa: E402


class _Blocked(RuntimeError):
    """Raised in place of a network call, so a miss is unmistakable and costs nothing."""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report Stage 3 extraction-cache coverage without calling any LLM")
    ap.add_argument("--sample-id", default=None,
                    help="conversation to probe (default: the first in the dataset)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="probe only the first N turns, for a quick answer")
    ap.add_argument("--no-dates", action="store_true",
                    help="match a run that was ingested with --no-dates")
    args = ap.parse_args()

    # The extractor logs an error for every blocked call; that is expected here and would
    # otherwise bury the report under hundreds of tracebacks.
    logging.getLogger("src.llm_extractor").setLevel(logging.CRITICAL)
    logging.getLogger("src.extractor").setLevel(logging.CRITICAL)

    conversations = load_conversations()
    conv = None
    for candidate in conversations:
        if args.sample_id is None or candidate.sample_id == args.sample_id:
            conv = candidate
            break
    if conv is None:
        have = ", ".join(c.sample_id for c in conversations[:12])
        print(f"unknown sample_id {args.sample_id!r}; have: {have}")
        return 2

    from src import config as c
    from src.extraction_cache import get_extraction_cache
    from src.extractor import MemoryExtractor

    cache = get_extraction_cache()
    print(f"conversation      : {conv.sample_id}")
    print(f"cache enabled     : {bool(cache.enabled)}"
          + ("" if cache.enabled else "   <-- probe is meaningless; see below"))
    print(f"cache directory   : {cache.directory}")
    try:
        on_disk = sum(1 for _ in cache.directory.glob("*/*.json"))
    except Exception:  # noqa: BLE001
        on_disk = -1
    print(f"entries on disk   : {on_disk}")
    print(f"model             : {c.LLM_EXTRACTION_MODEL}")
    print(f"temperature       : {c.STAGE_3_TEMPERATURE}   max_tokens: {c.STAGE_3_MAX_TOKENS}")
    print(f"cache version     : {c.EXTRACTION_CACHE_VERSION}")
    print()

    extractor = MemoryExtractor()
    if extractor.llm_extractor is None:
        print("STAGE 3 IS NOT AVAILABLE -- the LLM extractor failed to construct.")
        print("Extraction is running on Stage 1/2 regex only, which on narrative dialogue")
        print("stores very little, and the cache is never consulted at all.")
        print("Check the warning from src.extractor at startup for the reason.")
        return 1

    # Count hits at the cache itself rather than inferring them. A hit returns before
    # _call_llm is reached, and a miss falls through to it -- so hits plus blocked calls is
    # exactly the number of escalations, and it stays correct when the cache is disabled
    # (every escalation then shows up as a blocked call).
    counters = {"hit": 0, "blocked": 0}
    original_get = cache.get

    def counting_get(key: str):
        result = original_get(key)
        if result is None:
            return None
        counters["hit"] += 1
        return result

    def _blocked(prompt: str) -> str:
        counters["blocked"] += 1
        raise _Blocked("network disabled by cachecheck")

    cache.get = counting_get  # type: ignore[method-assign]
    extractor.llm_extractor._call_llm = _blocked  # type: ignore[method-assign]

    turns = conv.turns[:args.max_turns] if args.max_turns else conv.turns
    include_dates = not args.no_dates

    for i, turn in enumerate(turns, start=1):
        # Mirrors MemorySystem.process_turn exactly: turn_number starts at 1 and the
        # extractor's own context buffer fills as we go, so the prompts -- and therefore
        # the cache keys -- are the ones a real ingest would build.
        try:
            extractor.extract(
                turn.render(include_date=include_dates),
                i,
                speaker=turn.speaker,
                event_date=turn.session_date if include_dates else None,
                event_ts=turn.event_ts if include_dates else None,
            )
        except Exception:  # noqa: BLE001
            # extract() already swallows everything; this is belt and braces.
            pass

    hits, misses = counters["hit"], counters["blocked"]
    escalations = hits + misses
    coverage = hits / escalations if escalations else 0.0
    print(f"turns probed      : {len(turns)}")
    print(f"stage 3 escalations: {escalations}")
    print(f"served from cache : {hits}")
    print(f"would call the API: {misses}")
    print(f"coverage          : {coverage:.1%}")
    print()

    if escalations == 0:
        print("VERDICT: Stage 3 never escalated. Nothing would be cached or called.")
        return 1
    if coverage >= 0.98:
        print("VERDICT: warm. A re-ingest costs approximately nothing and rebuilds the")
        print("         same store. Safe to run --ingest-only.")
        return 0
    if hits == 0:
        print("VERDICT: cold. Every escalation would hit the API.")
        print("         Either the cache is disabled (set EXTRACTION_CACHE_ENABLED=true),")
        print("         it is pointed at a different EXTRACTION_CACHE_DIR, or the prompt/")
        print("         model/temperature/token-cap changed since it was populated -- any")
        print("         of which changes every key by design.")
        return 1
    print("VERDICT: partial. Only some turns are cached; the rest would call the API.")
    print("         Usually means the cache was populated by a run that stopped early.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
