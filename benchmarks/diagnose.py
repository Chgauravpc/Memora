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


# The worker writes per-question rows under "records" (worker.py). This module originally
# read "questions", which is not a key the worker produces -- so every run reported "no
# question records found" as though the benchmark had not been run, rather than as a
# mismatch. Both names are accepted, and an unrecognised payload now raises instead of
# quietly yielding nothing.
_RECORD_KEYS = ("records", "questions")


def load_records(results_dir: Path) -> List[Dict[str, Any]]:
    raw = results_dir / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"no results at {raw} - run the benchmark first")

    out: List[Dict[str, Any]] = []
    files = sorted(raw.glob("*.json"))
    seen_keys: set = set()

    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        seen_keys.update(payload.keys())
        rows = None
        for key in _RECORD_KEYS:
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            continue
        for rec in rows:
            rec.setdefault("_sample", payload.get("sample_id", f.stem))
            out.append(rec)

    if files and not out:
        # Distinguish "the benchmark produced nothing" from "this reader is looking in the
        # wrong place" -- they need completely different responses.
        raise ValueError(
            f"read {len(files)} results file(s) under {raw} but found no per-question "
            f"rows under any of {_RECORD_KEYS}.\n"
            f"Top-level keys present: {sorted(seen_keys)}\n"
            f"This is a reader/writer key mismatch, not an empty run."
        )
    return out


def audit_abstentions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find answers the judge credited even though the reader refused to answer.

    WHY THIS MATTERS MORE THAN ANY OTHER CHECK HERE

    On LoCoMo category 5 (adversarial) the gold answer says the information is absent, so
    a refusal IS the correct answer and the judge is instructed to accept it. On every
    other category the gold states a fact, and a refusal cannot be correct.

    So a non-adversarial question that was ABSTAINED and marked CORRECT is a judge error,
    and it inflates the headline score. A scorecard showing high abstention alongside a
    high judge score in the same category -- with token-F1 near zero -- is the exact
    signature. Since this check can only ever LOWER the reported number, it is worth
    running before quoting one.

    The reverse is also reported: adversarial questions where the reader correctly refused
    and the judge marked it wrong, which understates the score.
    """
    inflated: List[Dict[str, Any]] = []
    understated: List[Dict[str, Any]] = []
    per_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        cat = r.get("category")
        name = r.get("category_name", "?")
        abstained = bool(r.get("abstained"))
        correct = r.get("judge_correct")
        gold = (r.get("gold") or "").strip()

        per_cat[name]["n"] += 1
        if abstained:
            per_cat[name]["abstained"] += 1
        if correct:
            per_cat[name]["correct"] += 1

        if abstained and correct and cat != 5:
            per_cat[name]["abstained_but_correct"] += 1
            inflated.append(r)
        if cat == 5 and abstained and correct is False:
            understated.append(r)

    graded = [r for r in records if r.get("judge_correct") is not None]
    n_correct = sum(1 for r in graded if r.get("judge_correct"))
    adjusted = n_correct - len(inflated)

    return {
        "graded": len(graded),
        "correct": n_correct,
        "inflated": inflated,
        "understated": understated,
        "per_category": per_cat,
        "reported_score": (n_correct / len(graded)) if graded else None,
        # Lower bound: every credited refusal outside adversarial treated as wrong.
        "floor_score": (adjusted / len(graded)) if graded else None,
    }


def render_audit(a: Dict[str, Any], show: int = 4) -> str:
    L: List[str] = []
    L.append("=" * 70)
    L.append("Judge audit - refusals credited as correct")
    L.append("=" * 70)

    if not a["graded"]:
        L.append("no graded questions found")
        return "\n".join(L)

    L.append(f"graded questions : {a['graded']}")
    L.append(f"judged correct   : {a['correct']}")
    L.append("")
    L.append(f"  reported score            : {a['reported_score']:.1%}")
    L.append(f"  floor (credited refusals  : {a['floor_score']:.1%}")
    L.append(f"    outside adversarial = 0)")
    L.append("")
    L.append(f"  refusals credited outside adversarial : {len(a['inflated'])}")
    L.append(f"  adversarial refusals marked wrong     : {len(a['understated'])}")
    L.append("")

    L.append(f"    {'category':<16}{'n':>4}{'abstain':>9}{'correct':>9}{'abs+corr':>10}")
    for cat, c in sorted(a["per_category"].items()):
        L.append(f"    {cat:<16}{c['n']:>4}{c['abstained']:>9}{c['correct']:>9}"
                 f"{c.get('abstained_but_correct', 0):>10}")
    L.append("")

    if a["inflated"]:
        L.append("-" * 70)
        L.append("CREDITED REFUSALS  (gold states a fact; the reader refused; judge said correct)")
        L.append("-" * 70)
        for r in a["inflated"][:show]:
            L.append(f"  [{r.get('category_name')}] {r.get('question')}")
            L.append(f"    gold      : {r.get('gold')}")
            L.append(f"    predicted : {(r.get('prediction') or '')[:120]}")
            L.append("")
        L.append("  Each of these is a judge error that RAISES the reported score.")
        L.append("  Quote the floor, or fix the judge prompt and re-grade.")
    else:
        L.append("  No refusals were credited outside adversarial - the judge is not")
        L.append("  inflating the score through this path.")

    if a["understated"]:
        L.append("")
        L.append("-" * 70)
        L.append("ADVERSARIAL REFUSALS MARKED WRONG  (these LOWER the reported score)")
        L.append("-" * 70)
        for r in a["understated"][:show]:
            L.append(f"  {r.get('question')}")
            L.append(f"    gold      : {r.get('gold')}")
            L.append(f"    predicted : {(r.get('prediction') or '')[:120]}")
            L.append("")
        L.append("  On adversarial the gold says the information is absent, so a refusal")
        L.append("  should be CORRECT. If these look like proper refusals, the judge")
        L.append("  prompt is not recognising the gold as an absence statement.")
    return "\n".join(L)


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
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    if not records:
        print(f"no results files under {args.results / 'raw'} - run the benchmark first")
        return 1

    # The judge audit runs first and needs no saved contexts, so it works on any results
    # directory. It is the check most likely to reduce the headline number, which is
    # exactly why it should not be optional.
    audit = render_audit(audit_abstentions(records), show=args.show)
    print(audit)
    print()

    text = render(analyse(records, args.threshold), show=args.show, category=args.category)
    print(text)

    out = args.results / "diagnosis.txt"
    out.write_text(audit + "\n\n" + text, encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
