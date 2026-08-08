"""
Benchmark worker: replays ONE LoCoMo conversation through Memora, then answers its
questions.

Runs as its own OS process. That is not incidental -- src/config.py reads REDIS_DB and
QDRANT_COLLECTION into module-level constants at import time, so per-worker isolation is
only achievable by setting the environment before `src` is imported. A thread pool inside
one process would silently share Redis DB 0 and cross-contaminate every conversation.

Invoked by runner.py; also runnable directly for debugging a single conversation:

    REDIS_DB=1 QDRANT_COLLECTION=locomo_w1 \
      python -m benchmarks.worker --sample-id conv_0 --out results/locomo/raw/conv_0.json
"""

from __future__ import annotations

# Cache redirection MUST precede any transformers/sentence-transformers import.
from .paths import redirect_caches_into_repo, ensure_dirs  # noqa: E402
redirect_caches_into_repo()
ensure_dirs()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from .dataset import Conversation, Question, load_conversations  # noqa: E402
from .llm import LLMClient  # noqa: E402
from .paths import assert_contained  # noqa: E402
from . import qa as qa_mod  # noqa: E402

logger = logging.getLogger("benchmark.worker")


class Stage3Counter:
    """
    Counts Stage 3 LLM extraction calls by wrapping the extractor instance.

    `process_turn`'s stats dict has no field for this (memory_system.py:182) and the
    extractor only mentions it in a log line, so escalation rate -- the single biggest
    cost driver for this benchmark -- is otherwise unmeasurable. Wrapping the bound
    method keeps src/ untouched.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0

    def attach(self, memory_system: Any) -> bool:
        extractor = getattr(getattr(memory_system, "extractor", None), "llm_extractor", None)
        if extractor is None or not hasattr(extractor, "extract"):
            return False

        original = extractor.extract

        def counting_extract(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            try:
                return original(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                # A dead Stage 3 must not abort a multi-hour replay. Losing one turn's
                # LLM extraction degrades recall for that turn only; the count is
                # reported so the loss is visible rather than silent.
                self.failures += 1
                logger.warning("Stage 3 extraction failed: %s", exc)
                return []

        extractor.extract = counting_extract  # type: ignore[method-assign]
        return True


def _reset_user_dir(user_id: str) -> None:
    """
    Wipe this conversation's flat-file core memory.

    Phase 4 promotion writes into memory/<user_id>/*.md. Those files are ALWAYS injected
    verbatim, so leftovers from a previous conversation would leak one person's core
    memory into another's evaluation.
    """
    from src.config import MEMORY_DIR
    user_dir = Path(MEMORY_DIR) / user_id
    if user_dir.exists():
        shutil.rmtree(user_dir, ignore_errors=True)


def _architecture_snapshot() -> Dict[str, Any]:
    """Which memory-architecture mechanisms were active for this run.

    Imported lazily because src.config freezes env into module constants at import time,
    and the worker sets those constants up before this is called.
    """
    try:
        from src import config as c
    except Exception:  # noqa: BLE001
        return {"profile": "unknown"}
    return {
        "profile": getattr(c, "MEMORA_PROFILE", "unknown"),
        "ranking_weights": dict(getattr(c, "RANKING_WEIGHTS_5_SIGNAL", {})),
        "lexical_search": getattr(c, "LEXICAL_SEARCH_ENABLED", None),
        "chronological_context": getattr(c, "CONTEXT_CHRONOLOGICAL", None),
        "context_dates": getattr(c, "MEMORY_CONTEXT_INCLUDE_DATE", None),
        "context_speaker": getattr(c, "CONTEXT_INCLUDE_SPEAKER", None),
        "query_aware": getattr(c, "QUERY_AWARE_RETRIEVAL", None),
        "multihop_expansion": getattr(c, "MULTIHOP_EXPANSION_ENABLED", None),
        "dedup_speaker": getattr(c, "DEDUP_KEY_INCLUDES_SPEAKER", None),
        "dedup_value": getattr(c, "DEDUP_KEY_INCLUDES_VALUE", None),
        "embed_natural": getattr(c, "EMBED_NATURAL_TEXT", None),
        "max_memories": getattr(c, "MAX_MEMORIES_TO_RETRIEVE", None),
    }


def _stratified_sample(questions: List[Question], limit: int) -> List[Question]:
    """Take `limit` questions spread across categories, not the first `limit`.

    LoCoMo's question list is grouped, so a plain head-slice is badly unrepresentative:
    `--max-questions 20` on conversation 1 yields only multi-hop, temporal and open-domain
    -- it never reaches single-hop (the easiest category) or adversarial. A smoke test on
    that slice reports a score far below what the full run would give, and the difference
    is sampling, not the system.

    Round-robin across categories preserves category coverage at any limit.
    """
    if limit >= len(questions):
        return questions

    by_cat: Dict[int, List[Question]] = {}
    for q in questions:
        by_cat.setdefault(q.category, []).append(q)

    picked: List[Question] = []
    # Sorted for determinism: the same --max-questions must select the same questions on
    # every run, or resumed/repeated runs would not be comparable.
    cats = sorted(by_cat)
    idx = 0
    while len(picked) < limit:
        progressed = False
        for c in cats:
            bucket = by_cat[c]
            if idx < len(bucket):
                picked.append(bucket[idx])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return picked


def run_conversation(
    conv: Conversation,
    out_path: Path,
    include_adversarial: bool = True,
    max_questions: Optional[int] = None,
    include_dates: bool = True,
    progress_every: int = 100,
    save_context: bool = False,
    reuse_store: bool = False,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    from src import MemorySystem

    user_id = f"locomo_{conv.sample_id}"

    system = MemorySystem(user_id=user_id)
    health = system.health_check()

    counter = Stage3Counter()
    instrumented = counter.attach(system)

    # ------------------------------------------------------------- ingest
    #
    # Ingest is the expensive half (419 turns at ~0.76 s/turn, dominated by Stage 3 LLM
    # calls at ~80% escalation). When iterating on the READER or on diagnosis, re-ingesting
    # is pure waste: the store already holds the memories and nothing about ingest changed.
    # --reuse-store skips straight to QA, turning a ~6 minute loop into seconds.
    #
    # Only safe because each worker owns its own Redis DB and Qdrant collection, so the
    # store found here is necessarily this conversation's. It is opt-in because a stale
    # store silently measures the wrong thing -- if extraction or ingest changed, the
    # numbers would describe the previous code.
    existing = 0
    if reuse_store:
        try:
            existing = system.redis_store.count_memories()
        except Exception:  # noqa: BLE001
            existing = 0

    turns = conv.turns if max_turns is None else conv.turns[:max_turns]

    if reuse_store and existing > 0:
        print(f"[{conv.sample_id}] reusing existing store: {existing} memories, "
              f"skipping ingest of {len(turns)} turns", flush=True)
        ingest_start = time.time()
        turn_errors = 0
        extracted_total = 0
        turns = []
    else:
        if reuse_store:
            print(f"[{conv.sample_id}] --reuse-store requested but the store is empty; "
                  f"ingesting normally", flush=True)
        _reset_user_dir(user_id)
        # Isolation is per Redis DB + per Qdrant collection, so this clears only our slice.
        system.clear_memories()
        ingest_start = time.time()
        turn_errors = 0
        extracted_total = 0

    for i, turn in enumerate(turns, 1):
        try:
            # Speaker and date go through the API now rather than being folded into the
            # text and hoped for. Folding made them extraction-dependent: whether a fact
            # kept its attribution came down to whether the LLM happened to copy it into
            # the value. As real fields they are always present, indexable and rankable.
            _, stats = system.process_turn(
                turn.render(include_date=include_dates),
                speaker=turn.speaker,
                event_date=turn.session_date if include_dates else None,
                event_ts=turn.event_ts if include_dates else None,
            )
            extracted_total += int(stats.get("extracted_count", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            turn_errors += 1
            logger.warning("[%s] turn %d failed: %s", conv.sample_id, i, exc)
        if progress_every and i % progress_every == 0:
            rate = counter.calls / i if i else 0
            done = time.time() - ingest_start
            eta = (done / i) * (len(turns) - i)
            # print(), not logger.info(): .env sets LOG_LEVEL=WARNING for readable logs,
            # which silences INFO progress entirely. The run then looks hung for minutes
            # with no output at all -- worker stdout is redirected to logs/worker_*.log,
            # so the terminal shows nothing either. Progress must not be log-level gated.
            print(f"[{conv.sample_id}] ingest {i}/{len(turns)} turns "
                  f"({i / len(turns):.0%})  stage3 {rate:.0%}  "
                  f"{done / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                  flush=True)

    ingest_seconds = time.time() - ingest_start
    ingest_stage3_calls = counter.calls

    try:
        # count_memories lives on RedisStore, not MemorySystem. It counts the whole
        # Redis DB, which is exactly this conversation only because the worker owns
        # its own REDIS_DB.
        total_memories = system.redis_store.count_memories()
    except Exception:  # noqa: BLE001
        total_memories = -1

    # ---------------------------------------------------------------- QA
    client = LLMClient()
    questions: List[Question] = [q for q in conv.questions if q.gold is not None]
    if not include_adversarial:
        questions = [q for q in questions if q.category != 5]
    if max_questions:
        questions = _stratified_sample(questions, max_questions)

    records: List[Dict[str, Any]] = []
    qa_start = time.time()

    for j, q in enumerate(questions, 1):
        # get_prompt_context = retrieval WITHOUT extraction, so grading does not write
        # the questions themselves into the store. It returns a TUPLE despite annotating
        # `str` (memory_system.py:393) -- unpack, do not stringify.
        try:
            result = system.get_prompt_context(q.question)
            context, active = result if isinstance(result, tuple) else (result, [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] retrieval failed for q%d: %s", conv.sample_id, j, exc)
            context, active = "", []

        answer = qa_mod.read(client, context or "", q.question)
        gold = q.gold or ""
        verdict = qa_mod.judge(client, q.question, gold, answer.text)

        records.append({
            "question": q.question,
            "gold": gold,
            "prediction": answer.text,
            "category": q.category,
            "category_name": q.category_name,
            "evidence": q.evidence,
            "judge_correct": verdict,
            "reader_failed": answer.failed,
            "abstained": answer.abstained,
            "token_f1": qa_mod.token_f1(answer.text, gold),
            "exact_match": qa_mod.exact_match(answer.text, gold),
            "retrieved_count": len(active),
            "context_chars": len(context or ""),
        })

        if save_context:
            # The single most useful diagnostic when the score is low: it distinguishes
            # "retrieval never surfaced the fact" from "retrieval surfaced it and the
            # reader still abstained". Those have opposite fixes, and the aggregate
            # scorecard cannot tell them apart. Off by default -- on a full run this adds
            # ~3 kB per question.
            records[-1]["context"] = context or ""

        if progress_every and j % max(progress_every // 20, 1) == 0:
            print(f"[{conv.sample_id}] QA {j}/{len(questions)} questions "
                  f"({(time.time() - qa_start) / 60:.1f} min)", flush=True)

    qa_seconds = time.time() - qa_start

    payload: Dict[str, Any] = {
        "sample_id": conv.sample_id,
        "speakers": [conv.speaker_a, conv.speaker_b],
        "health": health,
        "config": {
            "redis_db": os.getenv("REDIS_DB", "0"),
            "qdrant_collection": os.getenv("QDRANT_COLLECTION", "memory_vectors"),
            "include_dates": include_dates,
            "include_adversarial": include_adversarial,
            "reader_model": client.model,
            "reader_provider": client.provider,
            # Which architecture produced this result. Recorded per-file so a number can
            # never be quoted without knowing what was switched on -- these mechanisms
            # change the system under test, and a results file that does not say so is
            # not reportable.
            **_architecture_snapshot(),
        },
        "ingest": {
            # Turns actually processed, not the conversation length: --max-turns and
            # --reuse-store both make these differ, and reporting the full length would
            # understate sec/turn and misstate the escalation denominator.
            "turns": len(turns),
            "turns_available": len(conv.turns),
            "reused_store": bool(reuse_store and existing > 0),
            "sessions": conv.num_sessions,
            "seconds": round(ingest_seconds, 1),
            "seconds_per_turn": round(ingest_seconds / max(len(conv.turns), 1), 3),
            "turn_errors": turn_errors,
            "extracted_total": extracted_total,
            "stage3_calls": ingest_stage3_calls,
            "stage3_rate": round(ingest_stage3_calls / max(len(conv.turns), 1), 4),
            "stage3_failures": counter.failures,
            "stage3_instrumented": instrumented,
            "memories_in_store": total_memories,
        },
        "qa": {
            "count": len(records),
            "seconds": round(qa_seconds, 1),
            "reader_judge_usage": client.usage.as_dict(),
        },
        "records": records,
    }

    out_path = assert_contained(out_path, "worker output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one LoCoMo conversation through Memora")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--no-dates", action="store_true",
                    help="Do not prefix session dates onto turns (ablation)")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Sample this many questions, spread across categories")
    ap.add_argument("--save-context", action="store_true",
                    help="Record the retrieved memory context per question (diagnosis)")
    ap.add_argument("--reuse-store", action="store_true",
                    help="Skip ingest if this worker's store already holds memories. "
                         "For iterating on the reader/diagnosis without paying for "
                         "re-extraction. Do NOT use after changing extraction.")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Ingest only the first N turns (faster smoke tests; note this "
                         "makes later questions unanswerable, so scores drop)")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Memora's own logging is extremely chatty at INFO (a banner per turn).
    logging.getLogger("src").setLevel(logging.WARNING)

    conversations = {c.sample_id: c for c in load_conversations()}
    if args.sample_id not in conversations:
        print(f"unknown sample_id {args.sample_id}; have: {list(conversations)[:12]}")
        return 2

    payload = run_conversation(
        conversations[args.sample_id],
        out_path=args.out,
        include_adversarial=not args.no_adversarial,
        max_questions=args.max_questions,
        include_dates=not args.no_dates,
        save_context=args.save_context,
        reuse_store=args.reuse_store,
        max_turns=args.max_turns,
    )
    ing = payload["ingest"]
    print(f"[{args.sample_id}] {ing['turns']} turns in {ing['seconds']}s "
          f"(stage3 {ing['stage3_rate']:.1%}), {payload['qa']['count']} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
