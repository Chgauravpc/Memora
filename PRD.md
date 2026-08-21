# Memora — Product Requirements Document

**Status:** Draft v1.0 · **Date:** 9 August 2026 · **Owner:** Gaurav Chaudhari

A long-form memory system for AI agents. This document specifies what Memora is, what it
must do, how it is measured, and what is currently wrong with it.

It describes the system **as built**, not as aspired to. Where the implementation and the
existing prose docs disagree, this document follows `src/config.py`.

---

## 1. Problem

An LLM has no memory between turns. The usual fix — replay the transcript — fails on three
counts as a conversation grows:

- **Context windows are finite.** A year of daily conversation does not fit, at any size.
- **Cost scales with tokens.** Replaying 200k tokens per turn is untenable per-request.
- **Relevance decays with length.** Models attend poorly to the middle of long contexts, so
  a fact buried at turn 400 is present but not usable.

What is needed is not a bigger window but **selection**: decide what is worth keeping, and
retrieve only what this turn needs.

**Memora's thesis:** most conversational turns contain nothing worth remembering. A cheap
cascade should discard those, an expensive model should handle the rest, and retrieval
should surface a small, ranked, budgeted set. The measurable claim is that a few thousand
tokens of *selected* memory beats a much larger window of raw transcript.

## 2. Goals and non-goals

### Goals

| # | Goal |
|---|---|
| G1 | Recall a fact stated hundreds of turns earlier, without the transcript |
| G2 | Keep per-turn cost roughly constant as conversation length grows |
| G3 | Handle corrections — later statements supersede earlier ones |
| G4 | Degrade to a weaker mode, never crash, when a backend is unavailable |
| G5 | Be measurable against a published third-party benchmark |
| G6 | Support multi-party, time-anchored conversation, not just a single first-person user |

### Non-goals

- **Not a chatbot.** Memora never generates responses. `process_turn` returns context; the
  caller calls the LLM. This boundary is deliberate and load-bearing.
- **Not a server.** Python library. No HTTP layer, no auth, no multi-tenancy story.
- **Not a RAG system over documents.** The unit of memory is a conversational fact, not a
  chunk of a corpus.
- **Not a vector database.** Qdrant is a dependency, not the product.

## 3. Users

| User | Needs |
|---|---|
| **Agent developer** (primary) | Drop-in memory for an assistant; one call per turn; no infra to design |
| **Researcher** | Ablatable components, reproducible numbers, honest baselines |
| **End user** (indirect) | An assistant that remembers, and updates when corrected |

## 4. Current state

Five feature layers, all implemented, stacked in the same modules rather than separated by
directory. "Phase" is the codebase's own vocabulary.

| Phase | Adds | Status |
|---|---|---|
| 1 | Flat-file core memory, Redis long-term store, heuristic + regex extraction | Working |
| 2 | Qdrant vectors, sentence-transformers embeddings, 3-signal ranking | Working |
| 3 | LLM extraction, semantic dedup, superseding, confidence modifiers | Working |
| 4 | Consolidation worker (decay / merge / promote), 5-signal ranking | Working; decay **disabled** |
| 5 | `evaluation/` RAGAS-style harness | Working, synthetic data only |
| — | LoCoMo benchmark harness (`benchmarks/`) | Working; first result 5%, diagnosed |
| — | Conversation architecture (`MEMORA_PROFILE`) | Implemented, **not yet measured** |

**Maturity: research prototype.** Correct enough to study, not hardened enough to deploy —
see §9.

## 5. Functional requirements

### 5.1 Ingestion — `process_turn`

Single entry point. Runs extract → store → dedup → retrieve → format and returns
`(memory_context: str, stats: dict)`.

- **FR-1.1** Accept a turn of text and return injectable context plus per-turn statistics.
- **FR-1.2** Accept optional `speaker`, `event_date`, `event_ts`. Absent ⇒ single-user
  behaviour unchanged.
- **FR-1.3** Never raise on a malformed turn; count and continue.
- **FR-1.4** Run consolidation inline every `CONSOLIDATION_INTERVAL_TURNS` (50).

### 5.2 Extraction — three-stage cascade

Each stage runs only if the previous was inconclusive. The economics of the system rest on
most turns never reaching stage 3.

- **FR-2.1 Stage 1 — heuristic filter.** Weighted length / keyword / question / specificity
  score against `SENSORY_FILTER_THRESHOLD` (0.3). Rejects greetings and sub-5-character
  messages outright.
- **FR-2.2 Stage 2 — regex classifier.** Pattern tables per memory type plus a
  payment/financial block. Returns typed `(key, value, confidence)` records.
- **FR-2.3 Stage 3 — LLM extraction.** Escalate when stage 2 returns nothing (and heuristic
  > 0.5) or its best confidence < `STAGE_3_CONFIDENCE_THRESHOLD` (0.7). Structured JSON,
  capped at `STAGE_3_MAX_TOKENS` (500).
- **FR-2.4** Merge stage 2 and 3 by `(type, key)`; higher confidence wins.
- **FR-2.5** Discard below `MIN_CONFIDENCE_TO_STORE` (0.6).
- **FR-2.6** Support Groq / OpenAI / Anthropic behind lazy per-provider clients, with N-key
  rotation and exponential backoff on 429.

**Memory types:** `preference`, `constraint`, `entity`, `instruction`, `commitment`,
`fact`, `event`.

> **Known gap.** Stage 2's patterns are first-person assistant phrasing ("I prefer…",
> "my name is…"). On third-person narrative they rarely fire: escalation measured **80.4%**
> on LoCoMo against **13.3%** in-domain. Cost impact is modest; the design assumption that
> stage 3 is exceptional does not hold off-domain.

### 5.3 Storage — four layers, three backends

| Layer | Backend | Role |
|---|---|---|
| **Core memory** | Markdown files, `memory/<user_id>/` | Always injected verbatim. Human-editable. Promotion target. |
| **Long-term** | Redis hashes `mem:<id>` | Authoritative record. Indices: `type:<type>`, `dedup:<…>`, `recent_memories`. |
| **Vector** | Qdrant `memory_vectors`, 384-dim cosine | Search only. **Not authoritative** — hits hydrate from Redis. |
| **Lexical** | In-process BM25 | Search only. Rebuilt when store size changes. |

- **FR-3.1** Every memory carries: `memory_id`, `type`, `key`, `value`, `confidence`,
  `turn_number`, `timestamp`, `source_text`, `speaker`, `event_date`, `event_ts`, plus
  lifecycle fields.
- **FR-3.2** `event_date` is when the content *refers to*; `timestamp` is ingest wall-clock.
  These must not be conflated — on replayed conversations they differ by years.
- **FR-3.3** Exact-key dedup identity is `(type, key, speaker, hash(value))`.
- **FR-3.4** Redis cannot store `None` or `bool`; the write path coerces and the read path
  re-casts field by field. **Adding a field is a four-place change** (config, write,
  read-cast, vector payload).

### 5.4 Deduplication and superseding

- **FR-4.1** Semantic dedup at cosine ≥ `SEMANTIC_DEDUP_THRESHOLD` (0.92), same type and
  user, checking the top `SEMANTIC_DEDUP_CHECK_LIMIT` (5).
- **FR-4.2** Duplicate ⇒ boost the existing memory's confidence by
  `CONFIDENCE_BOOST_PER_MENTION` (0.1), capped at `MAX_CONFIDENCE` (0.95).
- **FR-4.3** `is_update` ⇒ store the new memory and mark the old `superseded_by`.
- **FR-4.4** Do **not** merge near-duplicates carrying different `event_date` values — two
  similar events on different dates are two events.
- **FR-4.5** Superseded memories are excluded from retrieval but retained.

### 5.5 Retrieval

Hybrid, multi-branch, then ranked.

- **FR-5.1 Dense branch.** Vector search, `SEMANTIC_SEARCH_LIMIT` (100) candidates above
  `MIN_SEMANTIC_SCORE` (0.1).
- **FR-5.2 Lexical branch.** BM25, fused with dense by Reciprocal Rank Fusion (`RRF_K` 60).
- **FR-5.3 Recency branch.** `RECENCY_RETRIEVAL_LIMIT` (50) recent memories, **bypassing**
  the similarity floor, scored `RECENCY_FALLBACK_SEMANTIC_SCORE` (0.15). This is what
  produces long-distance recall.
- **FR-5.4 Always-on types.** `constraint` and `instruction` are always candidates.
- **FR-5.5 Multi-hop expansion.** Optional second pass seeded with first-pass content;
  candidates enter at 0.6× score.
- **FR-5.6 Ranking.** Weighted sum over five signals — semantic, type, recency, frequency,
  confidence — weights profile-dependent and env-overridable.
- **FR-5.7 Query-aware adjustment.** Additive boosts for temporal intent and speaker match,
  derived from question *shape* only.
- **FR-5.8 Budget.** Top `MAX_MEMORIES_TO_RETRIEVE` (50), trimmed to `MEMORY_TOKEN_BUDGET`
  (3000). Core memory has its own `CORE_MEMORY_TOKEN_BUDGET` (500) and is always injected.

### 5.6 Consolidation

Runs inline on the ingest thread every 50 turns. Each operation independently gated.

- **FR-6.1 Decay** — reduce confidence with age/inactivity. **Currently disabled**
  (`MEMORY_DECAY_ENABLED = False`).
- **FR-6.2 Merge** — combine near-duplicates above `MERGE_SIMILARITY_THRESHOLD` (0.85),
  same type only.
- **FR-6.3 Promote** — copy high-value memories into core Markdown. Requires confidence
  ≥ 0.90, ≥ 3 mentions, ≥ 5 accesses, ≥ 50 turns old, and a type in `PROMOTABLE_TYPES`.

> **Structural coupling.** Promotion writes into `##` section headers in the core Markdown
> templates. Renaming a header silently breaks promotion.

### 5.7 Architecture profiles

`MEMORA_PROFILE` selects a coherent set of defaults. Every mechanism is individually
overridable, and results files record which were active.

| | `conversation` (default) | `legacy` |
|---|---|---|
| Dedup identity | type+key+speaker+value | type+key |
| Context layout | chronological, speaker-attributed | grouped by type |
| Dates in context | yes | no |
| Lexical + RRF | yes | no |
| Embedding text | natural sentence | `key \| value \| type:` |
| Ranking (semantic/type) | 0.55 / 0.10 | 0.30 / 0.40 |

## 6. Evaluation requirements

**A memory system's only meaningful claim is recall under distance. That must be measured
on data the system was not tuned on.**

### 6.1 Primary benchmark — LoCoMo

Maharana et al., ACL 2024. 10 conversations, ~5,882 turns, 1,986 gradable questions across
five categories: multi-hop, temporal, open-domain, single-hop, adversarial.

- **ER-1** Replay each conversation turn-by-turn through the real `process_turn`.
- **ER-2** Answer questions using retrieval only (`get_prompt_context`), never re-ingesting
  the question.
- **ER-3** Primary metric **LLM-as-judge**, for comparability with published Mem0/Zep
  numbers. Report token-F1 and exact-match as deterministic secondaries.
- **ER-4** Score per category. An aggregate alone hides the failure mode.
- **ER-5** Isolate conversations by process, with a dedicated Redis DB and Qdrant collection
  each. Redis keys are global, so threads would cross-contaminate.
- **ER-6** Flag any run where the vector store was unreachable. Silent degradation to
  Phase 1 produces plausible, meaningless numbers.
- **ER-7** Record the active architecture profile in every results file.

### 6.2 Targets

| Milestone | Aggregate judge score | Status |
|---|---|---|
| First measurement | — | **5.0%** (biased sample, pre-redesign) |
| M1 — architecture fixes land | ≥ 25% | not measured |
| M2 — competitive | ≥ 40% | not measured |
| M3 — comparable to published systems | ≥ 55% | aspirational |

Published LoCoMo numbers cluster in the 50–70% band depending on judge and configuration.
**Cross-system comparison is only valid with the same judge model and prompt**, which is why
the harness fixes both.

### 6.3 Secondary evaluation

- **ER-8** Distance sweep — recall as a function of turns since the fact was stated, against
  a store populated with realistic distractors (**not** filler that stage 1 rejects).
- **ER-9** Ablations: dates on/off, lexical on/off, ranking profiles, consolidation on/off,
  top-K sweep.
- **ER-10** `benchmarks/selftest.py` must pass — logic checks with no backend, no network.

> **Retired claim.** "100% recall at 1000 turns" is not evidence. The store held ~5 real
> memories against a top-K of 50; filler turns were rejected at stage 1. The result was
> arithmetically forced and must not be cited.

## 7. Non-functional requirements

| ID | Requirement | Current |
|---|---|---|
| NFR-1 | Non-LLM work < 100 ms/turn | ~90 ms |
| NFR-2 | End-to-end < 1 s/turn including stage 3 | 0.76 s measured |
| NFR-3 | Retrieval < 300 ms | ~294 ms |
| NFR-4 | Per-turn cost independent of conversation length | Holds — retrieval is top-K bounded |
| NFR-5 | Backend loss ⇒ documented degradation, never a crash | Holds; **too quiet** (§9) |
| NFR-6 | Memory footprint ~1 GB per worker process | Holds (torch + MiniLM) |
| NFR-7 | Full LoCoMo run ≤ 2 h and ≤ $10 on a paid tier | $7.52 / ~0.7 h projected |

**Degradation ladder:** Qdrant or `sentence-transformers` unavailable ⇒ Phase 1 retrieval.
LLM key absent ⇒ stage 3 disabled, stages 1–2 continue. Redis unavailable ⇒ hard failure
(it is authoritative).

## 8. Architecture

```
                       process_turn(text, speaker?, event_date?)
                                      │
        ┌─────────────────────────────▼─────────────────────────────┐
        │ EXTRACT — 3-stage cascade                                 │
        │   1 heuristic filter  →  2 regex classifier  →  3 LLM     │
        │   each stage runs only if the prior was inconclusive      │
        └─────────────────────────────┬─────────────────────────────┘
                                      │ typed (key, value, confidence)
        ┌─────────────────────────────▼─────────────────────────────┐
        │ STORE + DEDUP                                             │
        │   exact-key identity · semantic dedup · supersede         │
        └───────┬───────────────┬───────────────┬───────────────────┘
                │               │               │
          Redis (truth)   Qdrant (search)   BM25 (search)
                │               │               │
        ┌───────▼───────────────▼───────────────▼───────────────────┐
        │ RETRIEVE — dense ∪ lexical (RRF) ∪ recency ∪ always-on    │
        │   → 5-signal rank → query-aware boost → top-K → budget    │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
        ┌─────────────────────────────▼─────────────────────────────┐
        │ COMPOSE — core memory (always) + retrieved (chronological)│
        └─────────────────────────────┬─────────────────────────────┘
                                      ▼
                        (memory_context, stats)   →   caller's LLM

        every 50 turns, inline: CONSOLIDATE — decay · merge · promote
```

**Design decisions worth defending:**

1. **Redis is authoritative, not Qdrant.** Vector payloads are a cache; every hit hydrates
   from Redis. An embedding index should never be the system of record.
2. **Consolidation is inline, not threaded.** Predictable and debuggable; the cost is a
   latency spike every 50 turns.
3. **Configuration is centralised.** `src/config.py` is the single source of truth —
   nothing is hardcoded at call sites.
4. **The library never generates text.** Keeps the system independently measurable.

## 9. Known defects and technical debt

Ordered by severity. These are real and reproducible, not hypothetical.

| ID | Severity | Defect |
|---|---|---|
| **D1** | **High** | **Redis is not user-namespaced.** `mem:`, `type:`, `dedup:`, `recent_memories` are global. Two `MemorySystem` instances with different `user_id`s share one store; `count_memories()` is global; **`clear_memories()` wipes every user and drops the whole Qdrant collection.** Only flat files and the Qdrant payload filter isolate users. |
| **D2** | **High** | **Silent degradation is too quiet.** Qdrant down ⇒ Phase 1 retrieval with only a log warning. A "recall regression" is often a backend that failed to connect. Needs a loud, queryable health state. |
| **D3** | Medium | **Stage 2 is domain-specific.** First-person assistant phrasing only. 80.4% escalation off-domain versus a 13.3% design assumption. |
| **D4** | Medium | **Access counts double-increment** — once in `retriever`, again in `_compose_prompt_context`. Inflates the frequency signal (weight 0.05, so impact is small). Fix the duplication, do not compensate in the weights. |
| **D5** | Medium | **`get_prompt_context` returns a tuple**, not the `str` its annotation and docstring promise. |
| **D6** | Medium | **Prose docs have drifted from config.** `QUICK_REFERENCE.md`, `README.md`, `RESULTS_FEBRUARY_2026.md` state values that do not match the code, and reference scripts that do not exist. Read `config.py`. |
| **D7** | Low | **Memory decay disabled** (`MEMORY_DECAY_ENABLED = False`), so Phase 4 is only partly exercised. |
| **D8** | Low | **No unit-test suite.** `pytest` is a declared dependency; nothing is collected. `benchmarks/selftest.py` covers new logic only. |

## 10. Roadmap

**M0 — Measure the redesign.** *(next)*
Full LoCoMo run under both profiles; per-category scorecard; attribute the delta to each
mechanism. Gate: the conversation profile beats legacy, with the reason established by
`benchmarks.diagnose` rather than assumed.

**M1 — Close the diagnosed gaps.**
Whatever M0 identifies. Likely: extraction quality at high escalation, and whether a flat
50-memory context is the right shape.

**M2 — Correctness debt.** D1, D2, D4, D5. D1 is the blocker for any multi-user use.

**M3 — Broader evaluation.** LongMemEval adapter; distance sweep with real distractors;
publish ablations alongside the headline number.

**M4 — Productionisation.** Only after M2. Namespacing, health endpoint, a real test suite,
backpressure for consolidation.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Redesign does not move the score | High | Every mechanism independently ablatable; `diagnose` separates retrieval from reader failure |
| Extraction is the real ceiling | High | If the fact was never stored, no retrieval fixes it — measure store contents directly, not just scores |
| Judge variance across models | Medium | Fix judge model and prompt; report deterministic secondaries alongside |
| Free-tier rate limits block runs | Medium | Daily caps cannot be parallelised around; paid tier is ~$8, or self-host extraction |
| Benchmark overfitting | **High** | No dataset vocabulary in `src/`; no per-category branching; no tuning against gold answers. Enforced by review, not by tooling. |

## 12. Open questions

1. **Is `key: value` the right memory unit?** It is a lossy compression of an utterance.
   Evidence snippets are a partial answer; the question stands.
2. **Should retrieval return 50 flat memories?** A smaller, reranked, better-formatted set
   may beat a larger one. Untested.
3. **What is the right store size ceiling?** Nothing bounds growth with decay disabled.
4. **Should consolidation stay inline?** Fine at 50-turn intervals; unclear at scale.
5. **Is MiniLM-384 adequate?** Larger encoders are affordable on the target hardware. Never
   ablated.

---

### Appendix — key configuration

All in `src/config.py`.

| Constant | Value |
|---|---|
| `SENSORY_FILTER_THRESHOLD` | 0.3 |
| `STAGE_3_CONFIDENCE_THRESHOLD` | 0.7 |
| `MIN_CONFIDENCE_TO_STORE` | 0.6 |
| `MAX_MEMORIES_TO_RETRIEVE` | 50 |
| `MEMORY_TOKEN_BUDGET` | 3000 |
| `CORE_MEMORY_TOKEN_BUDGET` | 500 |
| `MIN_SEMANTIC_SCORE` | 0.1 |
| `SEMANTIC_DEDUP_THRESHOLD` | 0.92 |
| `MERGE_SIMILARITY_THRESHOLD` | 0.85 |
| `CONSOLIDATION_INTERVAL_TURNS` | 50 |
| `MEMORY_DECAY_ENABLED` | **False** |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` (384-dim) |
