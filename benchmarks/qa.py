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

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .llm import LLMClient

READER_SYSTEM = """\
You are answering questions about a long-running conversation between two people, using \
only the MEMORY CONTEXT provided.

Rules:
- Answer from the MEMORY CONTEXT only. Do not use outside knowledge.
- Be terse: a word, a name, a date, or a short phrase. No explanation, no full sentences.
- If the memory context does not contain the answer, reply exactly: NO_ANSWER
"""

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
