"""
How many free API accounts does a fully-free LoCoMo run need?

The answer is governed by **tokens per day (TPD)**, not requests per minute. RPM throttles
you for seconds; a daily cap cannot be waited out or parallelised around -- it can only be
widened by adding accounts. Groq enforces limits at the ORGANIZATION level, so six keys
from six different friends are six independent quotas, while six keys from one account are
one quota.

Free-tier limits (Groq, verified 2026-08-18 -- re-check, these move):

    openai/gpt-oss-120b       30 RPM   1,000 RPD    8,000 TPM    200,000 TPD
    openai/gpt-oss-20b        30 RPM   1,000 RPD    8,000 TPM    200,000 TPD
    qwen/qwen3.6-27b          30 RPM   1,000 RPD    8,000 TPM    200,000 TPD

llama-3.3-70b-versatile and llama-3.1-8b-instant, this file's previous baseline, have been
removed from Groq's catalog entirely (404 model_not_found) -- do not reintroduce them.

Unlike the old 70B/8B split, the three current candidates share IDENTICAL free-tier
limits, so model choice no longer changes the account count. It changes cost and latency
instead: 120b and 20b are both reasoning models that spend an uninstrumented chunk of every
call's TPD on hidden chain-of-thought before the visible answer, so raw parameter count is
not the only thing to weigh -- a smaller reasoning model may cost less per call for a short
answer (judge verdict, reader answer) without losing any of the free-tier ceiling.

Usage:
    python -m benchmarks.free_tier --keys 6
    python -m benchmarks.free_tier --keys 6 --escalation 1.0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .estimate import (JUDGE_IN, JUDGE_OUT, READER_IN, READER_OUT, STAGE3_IN,
                       STAGE3_OUT, load_shape)


@dataclass
class Tier:
    key: str
    name: str
    rpm: int
    rpd: int
    tpm: int
    tpd: int


TIERS: Dict[str, Tier] = {
    "gpt-oss-120b-free": Tier("gpt-oss-120b-free", "openai/gpt-oss-120b (free)",
                              30, 1_000, 8_000, 200_000),
    "gpt-oss-20b-free": Tier("gpt-oss-20b-free", "openai/gpt-oss-20b (free)",
                             30, 1_000, 8_000, 200_000),
    "qwen3.6-27b-free": Tier("qwen3.6-27b-free", "qwen/qwen3.6-27b (free)",
                             30, 1_000, 8_000, 200_000),
    # No paid/dev tier is modeled here -- the old "70b-dev" numbers were specific to a
    # model that no longer exists on Groq. Verify live limits for your account before
    # assuming a paid tier changes this math.
}


@dataclass
class Role:
    name: str
    calls: float
    tokens: float


def roles_for(turns: int, questions: int, escalation: float) -> List[Role]:
    s3 = turns * escalation
    return [
        Role("extraction", s3, s3 * (STAGE3_IN + STAGE3_OUT)),
        Role("reader", questions, questions * (READER_IN + READER_OUT)),
        Role("judge", questions, questions * (JUDGE_IN + JUDGE_OUT)),
    ]


def accounts_needed(tokens: float, calls: float, tier: Tier) -> tuple[int, str]:
    """Accounts to finish within ONE day, and which limit binds."""
    by_tokens = math.ceil(tokens / tier.tpd) if tier.tpd else 0
    by_calls = math.ceil(calls / tier.rpd) if tier.rpd else 0
    if by_tokens >= by_calls:
        return max(by_tokens, 1), "tokens/day"
    return max(by_calls, 1), "requests/day"


def days_with(keys: int, tokens: float, calls: float, tier: Tier) -> float:
    d_tok = tokens / (tier.tpd * keys) if tier.tpd else 0
    d_call = calls / (tier.rpd * keys) if tier.rpd else 0
    return max(d_tok, d_call)


SPLITS = {
    "all-api": ["extraction", "reader", "judge"],
    "local-extraction": ["reader", "judge"],
    "judge-only": ["judge"],
    "fully-local": [],
}

SPLIT_NOTES = {
    "all-api": "Everything hosted. Nothing runs on your server.",
    "local-extraction": "Local model does Stage 3; reader + judge hosted.",
    "judge-only": "Local model does extraction AND reader; only the judge is hosted. "
                  "Keeps the one component where quality moves the headline score.",
    "fully-local": "No API at all. Free by construction, but the judge is then a local "
                   "model and grading is noisier.",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Free-tier account math for a fully-free LoCoMo run")
    ap.add_argument("--keys", type=int, default=6,
                    help="API keys you have, each from a SEPARATE account (default 6)")
    ap.add_argument("--escalation", type=float, default=0.70)
    ap.add_argument("--overhead", type=float, default=1.25,
                    help="safety multiplier for retries and longer-than-modelled answers "
                         "(default 1.25)")
    args = ap.parse_args()

    convs, turns, questions, real = load_shape()
    roles = {r.name: r for r in roles_for(turns, questions, args.escalation)}

    print("=" * 74)
    print("Free-tier account requirements for a fully-free LoCoMo run")
    print("=" * 74)
    if not real:
        print("!! dataset absent; approximate shape. Run benchmarks.download.\n")
    print(f"workload    : {turns:,} turns @ {args.escalation:.0%} escalation, "
          f"{questions:,} questions")
    print(f"you have    : {args.keys} keys (assumed {args.keys} separate accounts)")
    print(f"overhead    : x{args.overhead} applied to token totals")
    print()

    print("token cost by role")
    print("-" * 74)
    for r in roles.values():
        print(f"  {r.name:<12}{r.calls:>8,.0f} calls   {r.tokens:>12,.0f} tokens")
    print(f"  {'TOTAL':<12}{sum(r.calls for r in roles.values()):>8,.0f} calls   "
          f"{sum(r.tokens for r in roles.values()):>12,.0f} tokens")
    print()
    print("  The reader dominates: MEMORY_TOKEN_BUDGET = 3000 means ~3.5k input tokens")
    print("  per question. That single setting is why an all-API free run is hopeless.")
    print()

    for tier_key in ("gpt-oss-120b-free", "gpt-oss-20b-free", "qwen3.6-27b-free"):
        tier = TIERS[tier_key]
        print("=" * 74)
        print(f"{tier.name}   {tier.tpd:,} TPD / {tier.rpd:,} RPD per account")
        print("=" * 74)
        print(f"{'split':<18}{'tokens':>13}{'accts for 1d':>14}{'binds on':>14}"
              f"{'your ' + str(args.keys) + ' keys':>15}")
        print("-" * 74)
        for split, names in SPLITS.items():
            if not names:
                continue
            tokens = sum(roles[n].tokens for n in names) * args.overhead
            calls = sum(roles[n].calls for n in names)
            need, binds = accounts_needed(tokens, calls, tier)
            days = days_with(args.keys, tokens, calls, tier)
            verdict = f"{days:.1f} d" if days > 1.0 else "FITS in 1 day"
            print(f"{split:<18}{tokens:>13,.0f}{need:>14}{binds:>14}{verdict:>15}")
        print()

    # ---------------------------------------------------------------- recommendation
    tier = TIERS["gpt-oss-20b-free"]
    judge_tokens = roles["judge"].tokens * args.overhead
    judge_calls = roles["judge"].calls
    need, binds = accounts_needed(judge_tokens, judge_calls, tier)
    have_tpd = tier.tpd * args.keys

    print("=" * 74)
    print("RECOMMENDATION")
    print("=" * 74)
    model_label = tier.name.replace(" (free)", "")
    print("Run extraction and the reader on your local model; put ONLY the judge on")
    print(f"Groq's free {model_label}. The judge is by far the cheapest role in tokens")
    print("and the one where model quality most directly moves the number you intend")
    print("to publish. Note: judge_tokens below still assumes a non-reasoning output")
    print("size (JUDGE_OUT in estimate.py) -- a reasoning model's hidden chain-of-")
    print("thought is NOT counted here and will use more of the TPD budget than shown.")
    print()
    print(f"  judge needs   : {judge_tokens:,.0f} tokens "
          f"(incl. x{args.overhead} overhead)")
    print(f"  your capacity : {have_tpd:,} tokens/day across {args.keys} accounts")
    print(f"  accounts req. : {need}  (binds on {binds})")
    print()
    if args.keys >= need:
        slack = have_tpd / judge_tokens - 1
        print(f"  => Your {args.keys} keys are ENOUGH. Completely free, one day, "
              f"{slack:.0%} headroom.")
        if slack < 0.35:
            extra = max(0, math.ceil(judge_tokens * 1.5 / tier.tpd) - args.keys)
            print(f"     Headroom is thin; {extra} more key(s) would make it comfortable.")
    else:
        print(f"  => You need {need - args.keys} MORE key(s) from separate accounts.")
    print()
    print("  Zero keys also works: run the judge locally too. Free by construction,")
    print("  but disclose that grading used a local model.")
    print()
    print("  Do NOT chase enough free accounts to host the reader -- that would take")
    print(f"  {accounts_needed(roles['reader'].tokens * args.overhead, roles['reader'].calls, tier)[0]}"
          f" free {model_label} accounts. You own a 64-core AMX server; use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
