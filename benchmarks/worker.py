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
) -> Dict[str, Any]:
    from src import MemorySystem

    user_id = f"locomo_{conv.sample_id}"
    _reset_user_dir(user_id)

    system = MemorySystem(user_id=user_id)
    health = system.health_check()

    # Isolation is per Redis DB + per Qdrant collection, so this clears only our slice.
    system.clear_memories()

    counter = Stage3Counter()
    instrumented = counter.attach(system)

    # ------------------------------------------------------------- ingest
    ingest_start = time.time()
    turn_errors = 0
    extracted_total = 0

    for i, turn in enumerate(conv.turns, 1):
        try:
            _, stats = system.process_turn(turn.render(include_date=include_dates))
            extracted_total += int(stats.get("extracted_count", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            turn_errors += 1
            logger.warning("[%s] turn %d failed: %s", conv.sample_id, i, exc)
        if progress_every and i % progress_every == 0:
            rate = counter.calls / i if i else 0
            logger.info("[%s] ingested %d/%d turns (stage3 %.0f%%)",
                        conv.sample_id, i, len(conv.turns), rate * 100)

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

        if progress_every and j % max(progress_every // 4, 1) == 0:
            logger.info("[%s] answered %d/%d questions", conv.sample_id, j, len(questions))

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
        },
        "ingest": {
            "turns": len(conv.turns),
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
    )
    ing = payload["ingest"]
    print(f"[{args.sample_id}] {ing['turns']} turns in {ing['seconds']}s "
          f"(stage3 {ing['stage3_rate']:.1%}), {payload['qa']['count']} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
