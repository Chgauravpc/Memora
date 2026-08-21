"""
Cross-encoder reranking of retrieved memories.

WHY A SECOND SCORING STAGE

The 5-signal sum reduces "does this memory answer this question" to a single number --
a cosine similarity, or an RRF stand-in for one -- and then adds four signals (type,
recency, frequency, confidence) that never look at the query at all. A bi-encoder computes
the query vector and the memory vector independently and compares them afterwards, so no
part of the pipeline ever reads the two together.

That is exactly the judgement a cross-encoder makes. It concatenates query and document into
one input and is trained to score relevance directly, which is why retrieve-then-rerank is
the standard shape: the first stage buys recall cheaply, the second buys precision on the
handful of candidates that survive.

The failure this targets is specific and recorded. Across two full store rebuilds, the same
adversarial question failed the same way: the gold fact WAS retrieved, but ranked below a
plausible-but-wrong memory, so it never reached the promoted evidence section. Leading the
context with the top-ranked memories cannot fix that -- it promotes whatever ranking already
chose. The ranking itself has to change, and this changes it using the one signal the
existing formula structurally lacks.

COST

One forward pass over `(query, memory_text)` pairs for the top RERANK_CANDIDATES only.
`ms-marco-MiniLM-L-6-v2` is a 6-layer model; 50 pairs is tens of milliseconds on CPU and it
runs once per question, not once per turn -- so it touches query latency, never ingest. No
new dependency: `CrossEncoder` ships inside sentence-transformers, already required.

DEGRADATION

If sentence-transformers or the model is unavailable, `rerank` returns its input unchanged
and logs once. This mirrors how every other optional layer behaves, with one difference that
matters: a reranker that silently no-ops looks exactly like a reranker that did not help, so
`is_available()` is exposed for the harness to record rather than left to inference.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_model = None
_load_failed = False


def _get_model(model_name: str):
    """Lazily load the cross-encoder once per process."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder %s ...", model_name)
        _model = CrossEncoder(model_name)
        logger.info("Cross-encoder ready")
    except Exception as exc:  # noqa: BLE001
        # Logged at warning exactly once. A reranker that fails to load must not turn every
        # subsequent question into a stack trace.
        logger.warning("Cross-encoder unavailable (%s); reranking disabled: %s",
                       model_name, exc)
        _load_failed = True
        _model = None
    return _model


def is_available(model_name: str) -> bool:
    """Whether reranking can actually run. Recorded by the harness, not inferred."""
    return _get_model(model_name) is not None


def reset() -> None:
    """Drop the loaded model and the failure latch. For tests."""
    global _model, _load_failed
    _model = None
    _load_failed = False


def memory_pair_text(memory: Dict, max_chars: int = 512) -> str:
    """The text the cross-encoder judges against the query.

    `source_text` is included and placed last: extraction compresses a sentence into
    `key: value` and frequently drops the proper noun or qualifier the question turns on,
    while the original utterance still carries it. Truncation from the end therefore costs
    the least-structured part first.
    """
    parts: List[str] = []
    speaker = str(memory.get('speaker') or '').strip()
    key = str(memory.get('key') or '').strip()
    value = str(memory.get('value') or '').strip()
    event_date = str(memory.get('event_date') or '').strip()
    source = str(memory.get('source_text') or '').strip()

    head = f"{key}: {value}" if key and value else (value or key)
    if speaker:
        head = f"{speaker} - {head}"
    if event_date:
        head = f"[{event_date}] {head}"
    if head:
        parts.append(head)
    if source and source != value:
        parts.append(source)

    text = " | ".join(parts)
    return text[:max_chars]


def rerank(
    query: str,
    memories: Sequence[Dict],
    model_name: str,
    candidates: int = 50,
    weight: float = 0.7,
    score_field: str = "retrieval_score",
) -> List[Dict]:
    """Reorder `memories` by blending their existing score with a cross-encoder score.

    Only the first `candidates` entries are rescored; anything past that keeps its original
    position behind them, which is the usual retrieve-then-rerank contract.

    Both score sets are min-max normalised before blending because a cross-encoder logit is
    unbounded and roughly centred on zero while `retrieval_score` sits near [0, 1] -- adding
    them raw would let the logit's scale decide the outcome regardless of `weight`.

    Returns the input list unchanged if the model is unavailable, the query is empty, or
    there is nothing to reorder.
    """
    if not memories or not query or not query.strip():
        return list(memories)
    if weight <= 0.0:
        return list(memories)

    model = _get_model(model_name)
    if model is None:
        return list(memories)

    head = list(memories[:candidates])
    tail = list(memories[candidates:])

    pairs: List[Tuple[str, str]] = [(query, memory_pair_text(m)) for m in head]
    try:
        raw_scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-encoder scoring failed; keeping original order: %s", exc)
        return list(memories)

    ce_scores = [float(s) for s in raw_scores]
    base_scores = [float(m.get(score_field, 0.0) or 0.0) for m in head]

    ce_norm = _minmax(ce_scores)
    base_norm = _minmax(base_scores)

    for memory, ce_raw, ce_n, base_n in zip(head, ce_scores, ce_norm, base_norm):
        blended = weight * ce_n + (1.0 - weight) * base_n
        # Kept for diagnosis: whether reranking moved a memory, and by how much, is the only
        # way to attribute a score change to this stage rather than to something else.
        memory['rerank_score'] = ce_raw
        memory['pre_rerank_score'] = memory.get(score_field, 0.0)
        memory[score_field] = blended

    head.sort(key=lambda m: m.get(score_field, 0.0), reverse=True)

    # Anything not rescored ranks below everything that was, so a blended score in [0, 1]
    # cannot be undercut by an unrescored memory that happens to carry a larger raw value.
    for memory in tail:
        memory['pre_rerank_score'] = memory.get(score_field, 0.0)
        memory[score_field] = -1.0

    return head + tail


def _minmax(values: List[float]) -> List[float]:
    """Scale to [0, 1]; all-equal input maps to 0.5 so it contributes no ordering."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]
