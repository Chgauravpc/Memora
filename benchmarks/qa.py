"""
Reader and judge.

Memora returns memory context and never generates text, so a benchmark needs both halves
added on top:

  Reader : (memory context, question) -> short answer
  Judge  : (question, gold, prediction) -> correct / incorrect

Primary metric is LLM-as-judge, because that is what the published LoCoMo numbers from
Mem0/Zep use and comparability is the whole point of picking LoCoMo. Token-level F1 and
exact match are computed alongside as cheap, deterministic, reproducible secondaries --
if the judge and F1 disagree wildly, distrust the judge before the system.
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .llm import LLMClient

READER_SYSTEM_V1 = """\
You are answering questions about a long-running conversation between two people, using \
only the MEMORY CONTEXT provided.

Rules:
- Answer from the MEMORY CONTEXT only. Do not use outside knowledge.
- Be terse: a word, a name, a date, or a short phrase. No explanation, no full sentences.
- If the memory context does not contain the answer, reply exactly: NO_ANSWER
"""

# V2 exists because measurement said the reader, not retrieval, had become the bottleneck:
# 7 of 9 wrong answers had the gold answer present in the context (65% mean gold-word
# coverage) and the reader refused anyway. The clearest single case:
#
#   context : - [1:14 pm on 25 May, 2023] Melanie - camping trip: June 2023 (event)
#   question: When is Melanie planning on going camping?
#   answer  : NO_ANSWER          <- "June 2023" is verbatim in the line
#
# Three defects in V1 produced that, and each gets an explicit rule below.
#
#   1. V1 never describes the context format. The reader is handed a timeline of
#      "[date] Speaker - key: value (type)" rows plus optional `said:` quotes and has to
#      guess what any of it means.
#   2. V1 gives no date semantics. The bracketed date is when the statement was MADE; the
#      date being ASKED about is often inside the value ("camping trip: June 2023"). Two
#      dates in one row, with no stated relationship, reads as contradictory -- and a
#      cautious reader resolves contradictions by refusing.
#   3. V1's only instruction about uncertainty is when to say NO_ANSWER, with nothing
#      pushing the other way. That is a one-sided prompt, and it produced one-sided
#      behaviour.
#
# This is a change to the benchmark's answering component, not to the memory system, and
# must be reported alongside any score. BENCH_READER_PROMPT=v1 restores the original.
READER_SYSTEM_V2 = """\
You answer questions about a long-running conversation, using ONLY the MEMORY CONTEXT.

HOW TO READ THE CONTEXT
Each line is one remembered fact:
    - [date] Speaker - key: value (type)
and may be followed by `said: "..."`, the original sentence it came from.
  * `Speaker` is who said it.
  * The `[date]` is WHEN IT WAS SAID -- not necessarily when the thing happened.
  * A date inside the value (e.g. "camping trip: June 2023") is when that thing happens
    or happened. For "when" questions this is usually the answer, NOT the bracketed date.
  * `said:` quotes are the most reliable evidence; prefer them when they conflict with a
    compressed value.

HOW TO ANSWER
- Answer if the context supports an answer -- including when you must combine two facts,
  or when the wording differs from the question. Partial evidence still beats refusing.
- Attribute carefully. A fact belongs to the Speaker it is listed under. If the question
  asks about one person and the context only supports it for another, do not transfer it.
- Be terse: a word, a name, a date, or a short phrase. No explanation, no sentences.
- Reply exactly NO_ANSWER only when nothing in the context bears on the question. Do not
  refuse merely because the context is indirect, incomplete, or differently worded.
"""

# V3 = V2 minus the attribution prohibition.
#
# V2 scored 32% against V1's 64% on the same 25 questions, with abstentions DOUBLING from
# 8 to 16 -- the opposite of its intent. That comparison is confounded (the store was
# rebuilt at a nonzero extraction temperature, and mean gold-word coverage fell 65% -> 42%
# on its own), so V2 is not proven guilty. But within V2 there is exactly one instruction
# that ADDS a reason to refuse where V1 had none:
#
#     "If the question asks about one person and the context only supports it for
#      another, do not transfer it."
#
# That was my own addition, based on reasoning rather than measurement, and abstention
# rising is the specific behaviour it would produce. Deciding a question is unanswerable
# because its premise looks wrong is the judge's call, not the reader's -- a reader should
# report what the context supports and let grading settle the rest.
#
# V3 keeps the parts aimed at a demonstrated failure (the format is now explained; the
# bracketed date is the utterance date while the asked-about date sits in the value) and
# drops the part aimed at a hypothetical one. All three remain selectable so the A/B is
# reproducible.
READER_SYSTEM_V3 = """\
You answer questions about a long-running conversation, using ONLY the MEMORY CONTEXT.

HOW TO READ THE CONTEXT
Each line is one remembered fact:
    - [date] Speaker - key: value (type)
and may be followed by `said: "..."`, the original sentence it came from.
  * `Speaker` is who said it.
  * The `[date]` is WHEN IT WAS SAID -- not necessarily when the thing happened.
  * A date inside the value (e.g. "camping trip: June 2023") is when that thing happens
    or happened. For "when" questions this is usually the answer, NOT the bracketed date.
  * `said:` quotes are the most reliable evidence; prefer them when they conflict with a
    compressed value.

HOW TO ANSWER
- Give your best answer whenever the context supports one -- including when you must
  combine two facts, or when the wording differs from the question.
- Answering from indirect or partial evidence is expected and correct. Do not refuse
  because the context is incomplete, differently worded, or only implies the answer.
- Be terse: a word, a name, a date, or a short phrase. No explanation, no sentences.
- Reply exactly NO_ANSWER only when the context contains nothing at all on the subject.
"""

_READER_PROMPTS = {
    "v1": READER_SYSTEM_V1,
    "v2": READER_SYSTEM_V2,
    "v3": READER_SYSTEM_V3,
}
READER_PROMPT_VERSION = os.getenv("BENCH_READER_PROMPT", "v3").strip().lower()
READER_SYSTEM = _READER_PROMPTS.get(READER_PROMPT_VERSION, READER_SYSTEM_V3)

READER_TEMPLATE = """\
MEMORY CONTEXT
--------------
{context}

QUESTION
--------
{question}

Terse answer (or NO_ANSWER):"""

JUDGE_SYSTEM = """\
You grade a predicted answer against a gold answer for a conversational-memory benchmark.

Mark CORRECT when the prediction conveys the same information as the gold answer, even if \
worded differently, with different granularity, or with extra harmless detail. Dates, \
names and numbers must agree in substance.

Mark INCORRECT when it contradicts the gold answer, omits the key fact, or is a refusal \
where the gold answer states a fact.

Special case: when the gold answer indicates the information is absent, unknown, or not \
mentioned, then a prediction of NO_ANSWER or an equivalent refusal is CORRECT.

Reply with exactly one word: CORRECT or INCORRECT
"""

JUDGE_TEMPLATE = """\
QUESTION: {question}
GOLD ANSWER: {gold}
PREDICTED ANSWER: {prediction}

Verdict (CORRECT or INCORRECT):"""

NO_ANSWER = "NO_ANSWER"


# ------------------------------------------------------------------ string metrics

def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    p_tokens = _normalize(prediction).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


# --------------------------------------------------------------------- reader

@dataclass
class Answer:
    text: str
    abstained: bool
    failed: bool = False


def read(client: LLMClient, context: str, question: str, max_tokens: int = 128) -> Answer:
    raw = client.chat(
        user=READER_TEMPLATE.format(context=context or "(no memories retrieved)",
                                    question=question),
        system=READER_SYSTEM,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    if raw is None:
        return Answer(text="", abstained=False, failed=True)

    text = raw.strip()
    # Models often wrap the sentinel ("I must reply NO_ANSWER."); treat any occurrence
    # in a short reply as an abstention.
    abstained = NO_ANSWER in text.upper()
    if abstained:
        text = NO_ANSWER
    return Answer(text=text, abstained=abstained)


# ---------------------------------------------------------------------- judge

def judge(client: LLMClient, question: str, gold: str, prediction: str) -> Optional[bool]:
    """
    True/False verdict, or None if the judge call failed outright (so the report can
    distinguish "graded wrong" from "never graded").
    """
    if not prediction.strip():
        return False

    raw = client.chat(
        user=JUDGE_TEMPLATE.format(question=question, gold=gold, prediction=prediction),
        system=JUDGE_SYSTEM,
        max_tokens=8,
        temperature=0.0,
    )
    if raw is None:
        return None

    verdict = raw.strip().upper()
    if "INCORRECT" in verdict:
        return False
    if "CORRECT" in verdict:
        return True
    return None
