"""
Cost and wall-clock projection for a LoCoMo run.

Reads the ACTUAL dataset when present, so the numbers are derived from real turn and
question counts rather than remembered ballparks. Falls back to documented approximations
with a loud warning when the dataset has not been downloaded yet.

The headline conclusion for the dual-socket 64-core target box: CPU is not the constraint.
Embedding, Redis and
Qdrant work is a rounding error next to LLM API latency, and LoCoMo only has ~10
conversations, so there are only ~10 units of useful parallelism no matter how many cores
are available. Wall clock is set by API throughput -- provider tier and number of separate
accounts -- not by the machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from .paths import LOCOMO_JSON

# ---- per-call token shapes -------------------------------------------------------------
# Stage 3 extraction: prompt template + recent-turn context, capped at
# STAGE_3_MAX_TOKENS = 500 output (src/config.py:148).
STAGE3_IN, STAGE3_OUT = 800, 200
# Reader: MEMORY_TOKEN_BUDGET = 3000 of context (src/config.py:64) + question + system.
READER_IN, READER_OUT = 3500, 60
JUDGE_IN, JUDGE_OUT = 250, 5

# Measured on this codebase (RESULTS_FEBRUARY_2026.md:395-412): 575 ms mean processing per
# turn at 13.3% escalation, 294 ms mean retrieval.
NON_LLM_SECONDS_PER_TURN = 0.09   # CPU/Redis/Qdrant only, LLM excluded
LLM_CALL_SECONDS = 1.2            # round trip for a 70B call

# Fallbacks if the dataset is absent. LoCoMo's public 10-conversation eval set.
FALLBACK_CONVS = 10
FALLBACK_TURNS_PER_CONV = 600
FALLBACK_QUESTIONS = 1986


@dataclass
class Pricing:
    """USD per 1M tokens. Defaults are Groq llama-3.3-70b list pricing; override if your
    tier or provider differs -- these move and are not authoritative."""
    name: str = "groq/llama-3.3-70b"
    input_per_m: float = 0.59
    output_per_m: float = 0.79


PRICING = {
    "groq-70b": Pricing("groq/llama-3.3-70b", 0.59, 0.79),
    "gpt-4o-mini": Pricing("openai/gpt-4o-mini", 0.15, 0.60),
    "gpt-4o": Pricing("openai/gpt-4o", 2.50, 10.00),
    "sonnet": Pricing("anthropic/claude-sonnet", 3.00, 15.00),
}


def load_shape() -> tuple[int, int, int, bool]:
    """(conversations, turns, questions, from_real_dataset)"""
    if LOCOMO_JSON.exists():
        try:
            from .dataset import iter_answerable, load_conversations
            convs = load_conversations()
            turns = sum(len(c.turns) for c in convs)
            qs = sum(len(list(iter_answerable(c))) for c in convs)
            return len(convs), turns, qs, True
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not read dataset ({exc}); using fallbacks\n")
    return (FALLBACK_CONVS, FALLBACK_CONVS * FALLBACK_TURNS_PER_CONV,
            FALLBACK_QUESTIONS, False)


def estimate(
    workers: int,
    escalation: float,
    pricing_key: str = "groq-70b",
    rpm: Optional[float] = None,
    tpm: Optional[float] = None,
    rpd: Optional[float] = None,
) -> None:
    convs, turns, questions, real = load_shape()
    price = PRICING[pricing_key]

    if not real:
        print("!! dataset not found - using DOCUMENTED APPROXIMATIONS, not real counts.")
        print("   Run `python -m benchmarks.download` then re-run for exact numbers.\n")

    stage3_calls = turns * escalation
    reader_calls = questions
    judge_calls = questions
    total_calls = stage3_calls + reader_calls + judge_calls

    tok_in = stage3_calls * STAGE3_IN + reader_calls * READER_IN + judge_calls * JUDGE_IN
    tok_out = stage3_calls * STAGE3_OUT + reader_calls * READER_OUT + judge_calls * JUDGE_OUT
    cost = tok_in / 1e6 * price.input_per_m + tok_out / 1e6 * price.output_per_m

    # Only ~one worker per conversation is useful; extra workers idle.
    effective = min(workers, convs)

    # Serial time along one worker's critical path.
    per_conv_turns = turns / max(convs, 1)
    per_conv_questions = questions / max(convs, 1)
    serial_seconds = (
        per_conv_turns * NON_LLM_SECONDS_PER_TURN
        + per_conv_turns * escalation * LLM_CALL_SECONDS
        + per_conv_questions * 2 * LLM_CALL_SECONDS
    )
    rounds = -(-convs // effective)  # ceil
    latency_hours = serial_seconds * rounds / 3600

    print("=" * 68)
    print("LoCoMo run estimate")
    print("=" * 68)
    src = "actual dataset" if real else "approximation"
    print(f"shape ({src}): {convs} conversations, {turns:,} turns, "
          f"{questions:,} gradable questions")
    print(f"Stage 3 escalation assumed: {escalation:.0%}")
    print(f"workers: {workers} requested, {effective} useful "
          f"(capped by {convs} conversations)")
    print()
    print("LLM calls")
    print("-" * 68)
    print(f"  Stage 3 extraction : {stage3_calls:>10,.0f}")
    print(f"  reader             : {reader_calls:>10,.0f}")
    print(f"  judge              : {judge_calls:>10,.0f}")
    print(f"  total              : {total_calls:>10,.0f}")
    print()
    print("tokens & cost")
    print("-" * 68)
    print(f"  input              : {tok_in:>12,.0f}")
    print(f"  output             : {tok_out:>12,.0f}")
    print(f"  cost ({price.name}) : ${cost:,.2f}")
    print()
    print("wall clock")
    print("-" * 68)
    print(f"  latency-bound (no rate limit) : {latency_hours:.2f} h")

    if rpm or tpm or rpd:
        bounds = []
        if rpm:
            bounds.append(("requests/min", total_calls / rpm / 60))
        if tpm:
            bounds.append(("tokens/min", (tok_in + tok_out) / tpm / 60))
        if rpd:
            # A daily cap cannot be worked around by waiting less; it sets a floor in DAYS.
            bounds.append(("requests/day", total_calls / rpd * 24))
        for label, hours in bounds:
            print(f"  {label:<30}: {hours:.2f} h")
        limit_hours = max(h for _, h in bounds)
        binding = max(bounds, key=lambda b: b[1])[0]
        total = max(latency_hours, limit_hours)
        print()
        if limit_hours > latency_hours:
            days = f" ({total / 24:.1f} days)" if total > 24 else ""
            print(f"  => expect ~{total:.1f} h{days}, bound by {binding}")
            if binding == "requests/day":
                print("     A daily request cap cannot be parallelised around. This tier")
                print("     cannot run the benchmark; upgrade or add separate accounts.")
        else:
            print(f"  => expect ~{total:.1f} h, bound by call latency")
    else:
        print("  (pass --rpm / --tpm / --rpd for your provider tier to get the binding limit)")

    print()
    print("notes")
    print("-" * 68)
    print("  * CPU is not the bottleneck. Non-LLM work is ~90 ms/turn; on 64 cores with")
    print(f"    {effective} workers that is under 5% utilisation. Adding cores does nothing.")
    print(f"  * LoCoMo has only {convs} conversations, so parallelism caps at {convs}.")
    print("  * RAM: each worker loads its own sentence-transformers model + torch,")
    print(f"    roughly 0.8-1.2 GB. {effective} workers => ~{effective * 1.0:.0f}-"
          f"{effective * 1.2:.0f} GB. Check free RAM before raising --workers.")
    print("  * Groq rate limits are PER ACCOUNT. Extra keys on one account do not help;")
    print("    separate accounts do. This is the main lever on wall clock.")
    print("  * Escalation rate is the dominant cost variable and is UNKNOWN until you")
    print("    smoke-test. Re-run this with the measured rate from the smoke test.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate LoCoMo run cost and wall clock")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--escalation", type=float, default=0.70,
                    help="fraction of turns escalating to Stage 3 LLM (default 0.70; "
                         "13.3%% in-domain, ~100%% off-domain per RESULTS_FEBRUARY_2026)")
    ap.add_argument("--pricing", choices=sorted(PRICING), default="groq-70b")
    ap.add_argument("--rpm", type=float, default=None, help="provider requests/minute limit")
    ap.add_argument("--tpm", type=float, default=None, help="provider tokens/minute limit")
    ap.add_argument("--rpd", type=float, default=None, help="provider requests/DAY limit (free tiers)")
    args = ap.parse_args()

    if not 0.0 <= args.escalation <= 1.0:
        print("--escalation must be between 0 and 1")
        return 2

    estimate(args.workers, args.escalation, args.pricing, args.rpm, args.tpm, args.rpd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
