"""
Aggregate worker outputs into a LoCoMo scorecard.

Reports PER CATEGORY, not just an aggregate. The predicted failure profile is bimodal --
decent on single-hop fact/preference recall, poor on temporal and multi-hop -- and an
aggregate number hides exactly the thing worth knowing.

Also surfaces the operational numbers that decide whether a bigger run is affordable:
Stage 3 escalation rate, seconds/turn, and how many questions were lost to LLM failures
rather than answered wrongly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .dataset import CATEGORY_NAMES
from .paths import RAW_DIR, RESULTS_DIR, ensure_dirs


def load_raw(raw_dir: Path = RAW_DIR) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: could not read {path.name}: {exc}")
    return out


def aggregate(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "judge_correct": 0, "ungraded": 0, "f1": 0.0,
                 "em": 0, "abstained": 0, "reader_failed": 0, "retrieved": 0}
    )

    turns = sessions = stage3_calls = turn_errors = 0
    ingest_seconds = qa_seconds = 0.0
    stage3_failures = 0
    memories: List[int] = []
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0, "failures": 0}
    uninstrumented: List[str] = []
    degraded: List[str] = []

    for p in payloads:
        ing = p.get("ingest", {})
        turns += ing.get("turns", 0)
        sessions += ing.get("sessions", 0)
        stage3_calls += ing.get("stage3_calls", 0)
        stage3_failures += ing.get("stage3_failures", 0)
        turn_errors += ing.get("turn_errors", 0)
        ingest_seconds += ing.get("seconds", 0.0)
        qa_seconds += p.get("qa", {}).get("seconds", 0.0)
        if ing.get("memories_in_store", -1) >= 0:
            memories.append(ing["memories_in_store"])
        if not ing.get("stage3_instrumented", True):
            uninstrumented.append(p.get("sample_id", "?"))

        # A conversation that ran without the vector store silently drops to Phase 1
        # retrieval. Its score is not comparable and must be flagged, not averaged in.
        health = p.get("health", {})
        if health and not health.get("vector_store", True):
            degraded.append(p.get("sample_id", "?"))

        for key in usage:
            usage[key] += p.get("qa", {}).get("reader_judge_usage", {}).get(key, 0)

        for rec in p.get("records", []):
            cat = rec.get("category_name") or CATEGORY_NAMES.get(rec.get("category", -1), "unknown")
            b = by_cat[cat]
            b["n"] += 1
            verdict = rec.get("judge_correct")
            if verdict is None:
                b["ungraded"] += 1
            elif verdict:
                b["judge_correct"] += 1
            b["f1"] += rec.get("token_f1", 0.0)
            b["em"] += 1 if rec.get("exact_match") else 0
            b["abstained"] += 1 if rec.get("abstained") else 0
            b["reader_failed"] += 1 if rec.get("reader_failed") else 0
            b["retrieved"] += rec.get("retrieved_count", 0)

    categories: Dict[str, Any] = {}
    total_n = total_correct = total_graded = 0
    for cat, b in sorted(by_cat.items()):
        n = int(b["n"])
        graded = n - int(b["ungraded"])
        categories[cat] = {
            "n": n,
            "graded": graded,
            "ungraded": int(b["ungraded"]),
            # Accuracy is over GRADED questions; ungraded ones are reported separately so
            # judge failures cannot masquerade as either correct or incorrect.
            "judge_accuracy": round(b["judge_correct"] / graded, 4) if graded else None,
            "token_f1": round(b["f1"] / n, 4) if n else None,
            "exact_match": round(b["em"] / n, 4) if n else None,
            "abstain_rate": round(b["abstained"] / n, 4) if n else None,
            "reader_failures": int(b["reader_failed"]),
            "mean_retrieved": round(b["retrieved"] / n, 1) if n else None,
        }
        total_n += n
        total_correct += int(b["judge_correct"])
        total_graded += graded

    return {
        "conversations": len(payloads),
        "overall": {
            "questions": total_n,
            "graded": total_graded,
            "ungraded": total_n - total_graded,
            "judge_accuracy": round(total_correct / total_graded, 4) if total_graded else None,
        },
        "categories": categories,
        "operational": {
            "turns_ingested": turns,
            "sessions": sessions,
            "turn_errors": turn_errors,
            "stage3_calls": stage3_calls,
            "stage3_rate": round(stage3_calls / turns, 4) if turns else None,
            "stage3_failures": stage3_failures,
            "ingest_seconds": round(ingest_seconds, 1),
            "seconds_per_turn": round(ingest_seconds / turns, 3) if turns else None,
            "qa_seconds": round(qa_seconds, 1),
            "mean_memories_in_store": round(sum(memories) / len(memories), 1) if memories else None,
            "reader_judge_usage": usage,
        },
        "warnings": {
            "stage3_uninstrumented": uninstrumented,
            "vector_store_down": degraded,
        },
    }


def render(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    ov = summary["overall"]
    op = summary["operational"]

    lines.append("=" * 66)
    lines.append("LoCoMo scorecard - Memora")
    lines.append("=" * 66)
    lines.append(f"conversations : {summary['conversations']}")
    acc = ov["judge_accuracy"]
    lines.append(f"questions     : {ov['questions']} ({ov['graded']} graded, "
                 f"{ov['ungraded']} ungraded)")
    lines.append(f"OVERALL       : {acc:.1%}" if acc is not None else "OVERALL       : n/a")
    lines.append("")
    lines.append(f"{'category':<14}{'n':>6}{'judge':>9}{'tok-F1':>9}{'EM':>8}"
                 f"{'abstain':>9}{'retr':>7}")
    lines.append("-" * 66)
    for cat, c in summary["categories"].items():
        j = f"{c['judge_accuracy']:.1%}" if c["judge_accuracy"] is not None else "n/a"
        f1 = f"{c['token_f1']:.3f}" if c["token_f1"] is not None else "n/a"
        em = f"{c['exact_match']:.1%}" if c["exact_match"] is not None else "n/a"
        ab = f"{c['abstain_rate']:.1%}" if c["abstain_rate"] is not None else "n/a"
        rt = f"{c['mean_retrieved']:.1f}" if c["mean_retrieved"] is not None else "n/a"
        lines.append(f"{cat:<14}{c['n']:>6}{j:>9}{f1:>9}{em:>8}{ab:>9}{rt:>7}")

    lines.append("")
    lines.append("operational")
    lines.append("-" * 66)
    s3 = op["stage3_rate"]
    lines.append(f"  turns ingested      : {op['turns_ingested']}")
    lines.append(f"  Stage 3 escalation  : "
                 f"{s3:.1%}" if s3 is not None else "  Stage 3 escalation  : n/a")
    lines.append(f"  sec/turn            : {op['seconds_per_turn']}")
    lines.append(f"  ingest wall clock   : {op['ingest_seconds'] / 3600:.2f} h (summed over workers)")
    lines.append(f"  QA wall clock       : {op['qa_seconds'] / 3600:.2f} h (summed over workers)")
    lines.append(f"  mean store size     : {op['mean_memories_in_store']} memories")
    u = op["reader_judge_usage"]
    lines.append(f"  reader+judge tokens : {u['input_tokens']:,} in / "
                 f"{u['output_tokens']:,} out over {u['calls']:,} calls")
    lines.append(f"  LLM retries/failures: {u['retries']} / {u['failures']}")
    if op["turn_errors"]:
        lines.append(f"  ingest turn errors  : {op['turn_errors']}")
    if op["stage3_failures"]:
        lines.append(f"  stage3 failures     : {op['stage3_failures']}")

    warn = summary.get("warnings", {})
    if warn.get("vector_store_down"):
        lines.append("")
        lines.append("  !! vector store was DOWN for: "
                     f"{', '.join(warn['vector_store_down'])}")
        lines.append("     Those conversations ran on Phase 1 retrieval only and are")
        lines.append("     NOT comparable. Re-run them before quoting any number.")
    if warn.get("stage3_uninstrumented"):
        lines.append(f"  !! stage3 counter did not attach for: "
                     f"{', '.join(warn['stage3_uninstrumented'])} "
                     f"(escalation rate understated)")

    lines.append("")
    lines.append("Note: 'judge' is LLM-as-judge (primary, comparable to published")
    lines.append("numbers). tok-F1 and EM are deterministic secondaries -- if they")
    lines.append("diverge sharply from the judge, audit the judge first.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate LoCoMo results")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "scorecard.json")
    args = ap.parse_args()

    ensure_dirs()
    payloads = load_raw(args.raw_dir)
    if not payloads:
        print(f"no results found in {args.raw_dir}; run the benchmark first")
        return 1

    summary = aggregate(payloads)
    text = render(summary)
    print(text)

    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    txt_path = args.out.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"\nwrote {args.out}\nwrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
