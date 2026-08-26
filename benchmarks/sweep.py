"""
Offline retrieval sweep: compare many retrieval configurations without spending a single
LLM call.

WHY THIS EXISTS

The daily budget allows roughly one graded run. A graded run costs reader plus judge tokens
for every question, and an ingest on top if the store has to be rebuilt -- so evaluating
eight single-variable ablations the obvious way costs eight days, and the answer arrives
long after it was useful.

But every retrieval parameter -- top-K, BM25 k1, the ranking weights, the always-on floor,
reranking -- changes only WHICH memories reach the reader. That is decidable without a
reader. This module replays the questions against an already-populated store, measures
whether the gold answer's content actually reached the context and where it ranked, and
reports every configuration side by side. No reader, no judge, no ingest, no API key.

The intended loop is: sweep tens of configurations here for free, pick the two or three
worth grading, and spend the day's budget confirming them.

WHAT IT MEASURES, AND WHAT IT CANNOT

Two metrics, deliberately, because each is blind to something the other sees.

  coverage   Fraction of the gold answer's content words present anywhere in the rendered
             context. This is the same function benchmarks/diagnose.py uses to separate
             retrieval misses from reader misses, so a config that scores badly here will
             produce retrieval misses there.

             It is BIASED BY CONTEXT SIZE. More memories means more words means higher
             coverage, mechanically, whether or not the extra memories are useful. Never
             read it as "bigger K is better" -- read it as a floor: find the SMALLEST K
             whose coverage has not yet started to fall. Context size is reported next to
             it so that trade is visible rather than hidden.

  gold_rank  Position of the first retrieved memory that itself covers the gold answer,
             and the MRR over those positions. This one is size-independent: adding
             memories below the gold cannot improve it. It is the honest metric for
             anything that changes ORDER -- reranking, BM25 k1, the ranking weights -- and
             the one to trust when the two disagree.

Neither can see a reader miss. A configuration that puts the answer at rank 1 may still be
answered wrongly, and this module will call that a success. It bounds what retrieval can
deliver; it does not predict the score. Treat the output as a shortlist, never as a result.

USAGE

    # populate the store once (this is the expensive part; the extraction cache makes any
    # later re-ingest free and identical)
    python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context --force

    # then sweep, for free, as often as you like
    python -m benchmarks.sweep --grid default
    python -m benchmarks.sweep --grid topk
    python -m benchmarks.sweep --grid full --max-questions 50

Each configuration runs in its own subprocess, because src/config.py freezes environment
into module constants at import time -- the same reason the benchmark's workers are
processes rather than threads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.dataset import Conversation, Question, load_conversations
from benchmarks.diagnose import gold_coverage
from benchmarks.paths import RESULTS_DIR, redirect_caches_into_repo
from benchmarks.worker import _stratified_sample

# A memory "carries" the gold answer if it covers at least this much of it. Matches the
# threshold diagnose.py uses to call a failure a retrieval miss rather than a reader miss,
# so the two modules agree about what "the answer was there" means.
GOLD_PRESENT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------- grids
#
# One variable at a time from the shipped baseline, because a combined change that moves
# the number tells you nothing about which half did it. The combination arms at the end are
# there only to check the two best single changes are not redundant.

def _grid_topk() -> List[Dict[str, str]]:
    return [{"MAX_MEMORIES_TO_RETRIEVE": str(k)} for k in (5, 10, 15, 20, 30, 50)]


def _grid_rerank() -> List[Dict[str, str]]:
    return [
        {},
        {"RERANK_ENABLED": "true", "RERANK_WEIGHT": "0.5"},
        {"RERANK_ENABLED": "true", "RERANK_WEIGHT": "0.7"},
        {"RERANK_ENABLED": "true", "RERANK_WEIGHT": "1.0"},
    ]


def _grid_bm25() -> List[Dict[str, str]]:
    # k1 is currently 50, which effectively removes term-frequency saturation. With
    # source_text indexed, a long repetitive memory then scales linearly and can outrank a
    # short exact match. 1.2 and 1.5 are the textbook range.
    return [{"BM25_K1": v} for v in ("1.2", "1.5", "2.0", "10.0", "50.0")]


def _grid_recency() -> List[Dict[str, str]]:
    # On a replayed transcript "recent" only means "near the end of the conversation",
    # which is close to meaningless for a question about the whole history.
    return [
        {},
        {"RANK_W_RECENCY": "0.0"},
        {"RECENCY_RETRIEVAL_LIMIT": "10"},
        {"RANK_W_RECENCY": "0.0", "RECENCY_RETRIEVAL_LIMIT": "10"},
    ]


def _grid_always_on() -> List[Dict[str, str]]:
    # constraint/instruction are injected regardless of relevance. In this domain they are
    # rare and rarely the answer, so they mostly consume slots.
    return [{"ALWAYS_ON_SEMANTIC_FLOOR": v} for v in ("0.0", "0.05", "0.15", "0.5")]


def _grid_evidence() -> List[Dict[str, str]]:
    # Evidence snippets quote source_text for the top-ranked memories. Extraction is known
    # to compress away the detail questions ask about, and source_text is where it survives.
    return [
        {"CONTEXT_EVIDENCE_TOP_N": n, "CONTEXT_EVIDENCE_MAX_CHARS": c}
        for n, c in (("8", "220"), ("15", "220"), ("15", "450"), ("25", "450"))
    ]


def _grid_subject() -> List[Dict[str, str]]:
    # Only the RANKING half is visible here: rendering changes which words sit where, not
    # which memories are retrieved, so coverage and MRR cannot see it. The rendering half
    # needs a graded run.
    return [{}, {"SUBJECT_AWARE_RANKING": "true"}]


def _grid_default() -> List[Dict[str, str]]:
    """A curated single-variable pass -- the one to run first."""
    out: List[Dict[str, str]] = [{}]
    out += [{"MAX_MEMORIES_TO_RETRIEVE": str(k)} for k in (10, 15, 20, 30)]
    out += [{"RERANK_ENABLED": "true", "RERANK_WEIGHT": "0.7"},
            {"RERANK_ENABLED": "true", "RERANK_WEIGHT": "1.0"}]
    out += [{"BM25_K1": "1.5"}]
    out += [{"RANK_W_RECENCY": "0.0"}]
    out += [{"ALWAYS_ON_SEMANTIC_FLOOR": "0.0"}]
    out += [{"CONTEXT_EVIDENCE_TOP_N": "15", "CONTEXT_EVIDENCE_MAX_CHARS": "450"}]
    # The two changes most likely to help, together, to check they are not redundant.
    out += [{"MAX_MEMORIES_TO_RETRIEVE": "15", "RERANK_ENABLED": "true",
             "RERANK_WEIGHT": "0.7"}]
    out += [{"MAX_MEMORIES_TO_RETRIEVE": "15", "RERANK_ENABLED": "true",
             "RERANK_WEIGHT": "0.7", "BM25_K1": "1.5",
             "CONTEXT_EVIDENCE_TOP_N": "15", "CONTEXT_EVIDENCE_MAX_CHARS": "450"}]
    return out


def _grid_full() -> List[Dict[str, str]]:
    """Cross product of the two levers most likely to interact. Slower."""
    out: List[Dict[str, str]] = []
    for k in ("10", "15", "20", "50"):
        for rr in (None, "0.7", "1.0"):
            for k1 in ("1.5", "50.0"):
                cfg = {"MAX_MEMORIES_TO_RETRIEVE": k, "BM25_K1": k1}
                if rr:
                    cfg["RERANK_ENABLED"] = "true"
                    cfg["RERANK_WEIGHT"] = rr
                out.append(cfg)
    return out


GRIDS = {
    "default": _grid_default,
    "topk": _grid_topk,
    "rerank": _grid_rerank,
    "bm25": _grid_bm25,
    "recency": _grid_recency,
    "always_on": _grid_always_on,
    "evidence": _grid_evidence,
    "subject": _grid_subject,
    "full": _grid_full,
}


def label(cfg: Dict[str, str]) -> str:
    if not cfg:
        return "baseline"
    short = {
        "MAX_MEMORIES_TO_RETRIEVE": "k",
        "RERANK_ENABLED": "rerank",
        "RERANK_WEIGHT": "rw",
        "BM25_K1": "k1",
        "RANK_W_RECENCY": "w_rec",
        "RECENCY_RETRIEVAL_LIMIT": "rec_lim",
        "ALWAYS_ON_SEMANTIC_FLOOR": "floor",
        "SUBJECT_AWARE_RANKING": "subj_rank",
        "SUBJECT_AWARE_CONTEXT": "subj_ctx",
        "CONTEXT_EVIDENCE_TOP_N": "ev_n",
        "CONTEXT_EVIDENCE_MAX_CHARS": "ev_c",
    }
    parts = []
    for key, val in cfg.items():
        if key == "RERANK_ENABLED":
            continue
        parts.append(f"{short.get(key, key)}={val}")
    if cfg.get("RERANK_ENABLED") == "true" and "RERANK_WEIGHT" not in cfg:
        parts.append("rerank")
    return " ".join(parts) or "baseline"


# ---------------------------------------------------------------------------- child

def _no_op_access_counts(redis_store: Any) -> None:
    """Stop the sweep from mutating the store it is measuring.

    Retrieval increments access_count and last_accessed_turn for everything it returns.
    Those feed the frequency signal and the promotion thresholds, so sweeping fifty configs
    would leave the store measurably different from the one the graded run will use -- and
    each config would be scored against a store the previous config had already altered.
    Wrapping the method here keeps src/ untouched, the same approach Stage3Counter takes.
    """
    redis_store.increment_access_count = lambda *a, **k: None  # type: ignore[assignment]


def _store_turn_number(redis_store: Any) -> int:
    """Highest turn number in the store.

    Recency is scored against the current turn, and in a real run QA happens after ingest,
    so the retriever sees turn ~419 rather than 0. A fresh MemorySystem starts at 0, which
    would make every memory look brand new and silently change the ranking being measured.
    """
    try:
        memories = redis_store.get_all_memories(limit=100000)
    except Exception:  # noqa: BLE001
        return 0
    best = 0
    for mem in memories:
        try:
            best = max(best, int(mem.get("turn_number", 0) or 0))
        except (TypeError, ValueError):
            continue
    return best


def run_one(sample_id: Optional[str], max_questions: Optional[int],
            include_adversarial: bool) -> Dict[str, Any]:
    """Evaluate the CURRENT environment's configuration against the existing store."""
    redirect_caches_into_repo()

    from src.lexical_index import memory_text
    from src.memory_system import MemorySystem

    conversations = load_conversations()
    conv: Optional[Conversation] = None
    for candidate in conversations:
        if sample_id is None or candidate.sample_id == sample_id:
            conv = candidate
            break
    if conv is None:
        raise SystemExit(f"conversation {sample_id!r} not found in the dataset")

    questions: List[Question] = [q for q in conv.questions if q.gold is not None]
    if not include_adversarial:
        questions = [q for q in questions if q.category != 5]
    if max_questions:
        questions = _stratified_sample(questions, max_questions)

    system = MemorySystem(user_id=f"locomo_{conv.sample_id}")
    _no_op_access_counts(system.redis_store)

    store_size = system.redis_store.count_memories()
    if store_size <= 0:
        raise SystemExit(
            "the store is empty -- run the benchmark once to populate it before sweeping "
            "(the sweep never ingests; that is the point)"
        )
    system.turn_number = _store_turn_number(system.redis_store)

    per_cat: Dict[str, Dict[str, Any]] = {}
    coverages: List[float] = []
    ranks: List[Optional[int]] = []
    ctx_chars: List[int] = []
    retrieved_counts: List[int] = []

    started = time.time()
    for q in questions:
        memories = system.retriever.retrieve(q.question, system.turn_number)
        context = system.retriever.format_memories_for_prompt(memories)

        cov = gold_coverage(q.gold or "", context)
        # First retrieved memory that itself carries the answer. Unlike coverage, this
        # cannot be improved by returning more memories.
        rank: Optional[int] = None
        for idx, mem in enumerate(memories, start=1):
            mem_cov = gold_coverage(q.gold or "", memory_text(mem))
            if mem_cov is not None and mem_cov >= GOLD_PRESENT_THRESHOLD:
                rank = idx
                break

        if cov is not None:
            coverages.append(cov)
        ranks.append(rank)
        ctx_chars.append(len(context))
        retrieved_counts.append(len(memories))

        bucket = per_cat.setdefault(q.category_name, {"n": 0, "cov": [], "hits": 0})
        bucket["n"] += 1
        if cov is not None:
            bucket["cov"].append(cov)
        if cov is not None and cov >= GOLD_PRESENT_THRESHOLD:
            bucket["hits"] += 1

    found = [r for r in ranks if r is not None]
    mrr = sum(1.0 / r for r in found) / len(ranks) if ranks else 0.0
    mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
    mean_chars = sum(ctx_chars) / len(ctx_chars) if ctx_chars else 0.0

    return {
        "questions": len(questions),
        "store_size": store_size,
        "turn_number": system.turn_number,
        "mean_coverage": round(mean_cov, 4),
        # Share of questions whose answer reached the context at all. This is the number
        # diagnose.py would count as "not a retrieval miss".
        "present_rate": round(
            sum(1 for c in coverages if c >= GOLD_PRESENT_THRESHOLD) / len(coverages), 4
        ) if coverages else 0.0,
        "gold_found_rate": round(len(found) / len(ranks), 4) if ranks else 0.0,
        "mrr": round(mrr, 4),
        "median_gold_rank": _median(found),
        "mean_context_chars": int(mean_chars),
        # Coverage bought per 1k characters of context. Makes the size bias in `coverage`
        # explicit instead of letting a bigger K quietly win on volume.
        "coverage_per_kchar": round(mean_cov / (mean_chars / 1000.0), 4) if mean_chars else 0.0,
        "mean_retrieved": round(sum(retrieved_counts) / len(retrieved_counts), 2)
        if retrieved_counts else 0.0,
        "seconds": round(time.time() - started, 1),
        "by_category": {
            name: {
                "n": b["n"],
                "mean_coverage": round(sum(b["cov"]) / len(b["cov"]), 4) if b["cov"] else 0.0,
                "present_rate": round(b["hits"] / b["n"], 4) if b["n"] else 0.0,
            }
            for name, b in sorted(per_cat.items())
        },
    }


def _median(values: List[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ---------------------------------------------------------------------------- parent

def _spawn(cfg: Dict[str, str], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    env = dict(os.environ)
    env.update(cfg)
    # Isolation must match the run that built the store, or the sweep reads an empty DB.
    env["REDIS_DB"] = str(args.slot)
    env["QDRANT_COLLECTION"] = f"locomo_w{args.slot}"
    env.setdefault("MEMORA_PROFILE", "conversation")

    cmd = [sys.executable, "-m", "benchmarks.sweep", "--child"]
    if args.sample_id:
        cmd += ["--sample-id", args.sample_id]
    if args.max_questions:
        cmd += ["--max-questions", str(args.max_questions)]
    if args.no_adversarial:
        cmd.append("--no-adversarial")
    cmd += ["--slot", str(args.slot)]

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        print(f"  FAILED: {' | '.join(tail)}")
        return None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print("  FAILED: child produced no result line")
    return None


def render(rows: List[Tuple[str, Dict[str, Any]]]) -> None:
    if not rows:
        print("no configurations completed")
        return

    print()
    print("=" * 104)
    print("RETRIEVAL SWEEP -- no reader, no judge, no ingest. A shortlist, not a result.")
    print("=" * 104)
    first = rows[0][1]
    print(f"store: {first.get('store_size')} memories   "
          f"questions: {first.get('questions')}   "
          f"turn: {first.get('turn_number')}")
    print()
    header = (f"{'config':<44}{'cover':>7}{'present':>9}{'MRR':>7}"
              f"{'rank':>6}{'ctx':>8}{'cov/kc':>8}{'ret':>6}")
    print(header)
    print("-" * len(header))

    base = next((r for lbl, r in rows if lbl == "baseline"), None)
    for lbl, r in rows:
        rank = r.get("median_gold_rank")
        print(f"{lbl:<44}"
              f"{r['mean_coverage']:>7.3f}"
              f"{r['present_rate']:>9.1%}"
              f"{r['mrr']:>7.3f}"
              f"{('-' if rank is None else f'{rank:.0f}'):>6}"
              f"{r['mean_context_chars']:>8}"
              f"{r['coverage_per_kchar']:>8.3f}"
              f"{r['mean_retrieved']:>6.1f}")

    if base:
        print()
        print("Deltas vs baseline (MRR is the one to trust for ordering changes; coverage "
              "rises with context size on its own):")
        ranked = sorted(
            (r for r in rows if r[0] != "baseline"),
            key=lambda kv: kv[1]["mrr"] - base["mrr"], reverse=True,
        )
        for lbl, r in ranked[:8]:
            d_mrr = r["mrr"] - base["mrr"]
            d_cov = r["mean_coverage"] - base["mean_coverage"]
            d_ctx = r["mean_context_chars"] - base["mean_context_chars"]
            print(f"  {lbl:<42} MRR {d_mrr:+.3f}   coverage {d_cov:+.3f}   "
                  f"context {d_ctx:+d} chars")

    print()
    print("Next: grade the two or three best with a real run --")
    print("  python run_locomo.py --limit 1 --max-questions 25 --workers 1 "
          "--save-context --force <flags>")
    print("Coverage cannot see a reader miss, so a config that wins here can still lose "
          "on the graded run.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare retrieval configurations offline, with no LLM calls")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--grid", default="default", choices=sorted(GRIDS),
                    help="which set of configurations to try (default: default)")
    ap.add_argument("--configs", default=None,
                    help="path to a JSON list of env-var dicts, instead of a named grid")
    ap.add_argument("--sample-id", default=None,
                    help="conversation to evaluate (default: the first in the dataset)")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="stratified sample of questions, as in run_locomo.py")
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--slot", type=int, default=0,
                    help="worker slot whose store to read: REDIS_DB and locomo_w<slot> "
                         "(default 0, which is what --workers 1 uses)")
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    args = ap.parse_args()

    if args.child:
        result = run_one(args.sample_id, args.max_questions, not args.no_adversarial)
        print(json.dumps(result))
        return 0

    if args.configs:
        configs = json.loads(Path(args.configs).read_text(encoding="utf-8"))
    else:
        configs = GRIDS[args.grid]()

    print(f"sweeping {len(configs)} configurations against the store in "
          f"REDIS_DB={args.slot} / locomo_w{args.slot}")
    print("no LLM calls are made; this costs nothing but wall clock\n")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for i, cfg in enumerate(configs, 1):
        lbl = label(cfg)
        print(f"[{i}/{len(configs)}] {lbl} ...", flush=True)
        result = _spawn(cfg, args)
        if result is not None:
            result["config"] = cfg
            rows.append((lbl, result))

    render(rows)

    out_path = Path(args.out) if args.out else (RESULTS_DIR / "sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"grid": args.grid, "results":
                    [{"label": l, **r} for l, r in rows]}, indent=2),
        encoding="utf-8")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
