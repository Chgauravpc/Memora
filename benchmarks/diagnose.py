"""
Explain a low LoCoMo score: was it retrieval, or was it the reader?

The scorecard tells you Memora got 5% and abstained on 90% of temporal questions. It
cannot tell you WHY, and the two candidate causes need opposite fixes:

  * RETRIEVAL MISS -- the fact never made it into the memory context. Fixing this means
    extraction (was the fact ever stored?) or ranking (was it stored but out-ranked?).
  * READER MISS   -- the fact WAS in the context and the reader still failed or abstained.
    Fixing this means the reader prompt, the context format, or the answer parsing.

Chasing the wrong one wastes hours, and abstention rates look identical either way.

This module reads results written with `--save-context` and, for every wrong answer, checks
whether the gold answer's content words are present in the retrieved context. That is a
crude oracle -- it over-counts when a gold word appears incidentally, and under-counts when
the context implies the answer in different words -- so treat the split as a strong hint
about where to look, not as a measurement. The printed examples are the real payload.

    python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context
    python -m benchmarks.diagnose
    python -m benchmarks.diagnose --category 2 --show 5   # temporal, 5 examples
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import RESULTS_DIR

# Words that carry no retrieval signal. Gold answers are short, so a couple of stopwords
# matching would otherwise mark a total miss as "present".
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
    "is", "was", "are", "were", "be", "been", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "that", "this", "as", "by", "from", "had", "has", "have",
    "did", "does", "do", "not", "no", "yes", "s", "t",
}

_WORD = re.compile(r"[a-z0-9']+")


def _content_words(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) > 1]


def gold_coverage(gold: str, context: str) -> Optional[float]:
    """Fraction of the gold answer's content words that appear in the context."""
    words = _content_words(gold)
    if not words:
        return None
    ctx = (context or "").lower()
    hit = sum(1 for w in set(words) if w in ctx)
    return hit / len(set(words))


def load_records(results_dir: Path) -> List[Dict[str, Any]]:
    raw = results_dir / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"no results at {raw} - run the benchmark first")
    out: List[Dict[str, Any]] = []
    for f in sorted(raw.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for rec in payload.get("questions", []) or []:
            rec.setdefault("_sample", payload.get("sample_id", f.stem))
            out.append(rec)
    return out


def analyse(records: List[Dict[str, Any]], present_threshold: float = 0.5) -> Dict[str, Any]:
    have_ctx = [r for r in records if "context" in r]
    wrong = [r for r in have_ctx if not r.get("judge_correct")]

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in wrong:
        cov = gold_coverage(r.get("gold", ""), r.get("context", ""))
        r["_gold_coverage"] = cov
        if cov is None:
            buckets["ungradable"].append(r)
        elif cov >= present_threshold:
            # The answer was retrievable and the system still missed it.
            buckets["reader_miss" if not r.get("reader_failed") else "reader_error"].append(r)
        else:
            buckets["retrieval_miss"].append(r)

    per_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, rows in buckets.items():
        for r in rows:
            per_cat[r.get("category_name", "?")][name] += 1

    return {
        "total": len(records),
        "with_context": len(have_ctx),
        "wrong": len(wrong),
        "buckets": buckets,
        "per_category": per_cat,
        "mean_coverage_wrong": (
            round(sum(r["_gold_coverage"] for r in wrong
                      if r.get("_gold_coverage") is not None)
                  / max(1, sum(1 for r in wrong if r.get("_gold_coverage") is not None)), 3)
        ),
    }


def render(a: Dict[str, Any], show: int = 3, category: Optional[int] = None) -> str:
    L: List[str] = []
    L.append("=" * 70)
    L.append("Failure diagnosis - retrieval miss vs reader miss")
    L.append("=" * 70)

    if a["with_context"] == 0:
        L.append("")
        L.append("No saved contexts found. Re-run with --save-context:")
        L.append("    python run_locomo.py --limit 1 --max-questions 25 "
                 "--workers 1 --save-context")
        return "\n".join(L)

    b = a["buckets"]
    n_wrong = max(1, a["wrong"])
    L.append(f"questions with context : {a['with_context']}")
    L.append(f"wrong answers          : {a['wrong']}")
    L.append("")

    def line(key: str, label: str) -> None:
        n = len(b.get(key, []))
        L.append(f"  {label:<34} {n:>4}  ({n / n_wrong:.0%} of wrong)")

    line("retrieval_miss", "retrieval miss (fact absent)")
    line("reader_miss", "reader miss (fact WAS present)")
    line("reader_error", "reader errored")
    line("ungradable", "gold had no content words")
    L.append("")
    L.append(f"  mean gold-word coverage on wrong answers: {a['mean_coverage_wrong']:.0%}")
    L.append("")

    L.append("  by category:")
    L.append(f"    {'category':<16}{'retr-miss':>10}{'reader-miss':>13}{'error':>8}")
    for cat, counts in sorted(a["per_category"].items()):
        L.append(f"    {cat:<16}{counts.get('retrieval_miss', 0):>10}"
                 f"{counts.get('reader_miss', 0):>13}{counts.get('reader_error', 0):>8}")
    L.append("")

    L.append("-" * 70)
    L.append("INTERPRETATION")
    L.append("-" * 70)
    rm = len(b.get("retrieval_miss", []))
    dm = len(b.get("reader_miss", []))
    if rm > dm * 2:
        L.append("  Dominated by RETRIEVAL misses. The reader is not the problem - the")
        L.append("  facts are not reaching it. Look at extraction first (is the fact")
        L.append("  stored at all?), then ranking (stored but out-ranked?). Compare a")
        L.append("  failing question's evidence turns against the store.")
    elif dm > rm * 2:
        L.append("  Dominated by READER misses. Retrieval is surfacing the facts and the")
        L.append("  answer step is failing anyway. Look at the reader prompt, the context")
        L.append("  format, and how abstention is being triggered - not at ranking.")
    else:
        L.append("  Mixed. Both retrieval and the reader are contributing; fix retrieval")
        L.append("  first, since reader misses measured against a bad context are not")
        L.append("  a stable signal.")
    L.append("")

    for key, label in (("retrieval_miss", "RETRIEVAL MISSES"),
                       ("reader_miss", "READER MISSES")):
        rows = b.get(key, [])
        if category is not None:
            rows = [r for r in rows if r.get("category") == category]
        if not rows:
            continue
        L.append("=" * 70)
        L.append(f"{label} - {min(show, len(rows))} of {len(rows)}")
        L.append("=" * 70)
        for r in rows[:show]:
            ctx = (r.get("context") or "").replace("\n", " ")
            L.append(f"  [{r.get('category_name')}] {r.get('question')}")
            L.append(f"    gold       : {r.get('gold')}")
            L.append(f"    predicted  : {(r.get('prediction') or '')[:160]}")
            L.append(f"    abstained  : {r.get('abstained')}   "
                     f"retrieved: {r.get('retrieved_count')}   "
                     f"coverage: {r.get('_gold_coverage'):.0%}")
            L.append(f"    context    : {ctx[:400]}{'...' if len(ctx) > 400 else ''}")
            L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose whether low LoCoMo scores come from retrieval or the reader")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--show", type=int, default=3, help="examples per bucket")
    ap.add_argument("--category", type=int, default=None,
                    help="only show examples from this LoCoMo category (1-5)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="gold-word coverage above which the fact counts as retrieved")
    args = ap.parse_args()

    try:
        records = load_records(args.results)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    if not records:
        print("no question records found")
        return 1

    text = render(analyse(records, args.threshold), show=args.show, category=args.category)
    print(text)
    out = args.results / "diagnosis.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
