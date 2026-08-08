# Why the first LoCoMo run scored 5%

First smoke test, conversation 1, 20 questions:

```
OVERALL       : 5.0%
category           n    judge   tok-F1      EM  abstain   retr
multi_hop          8     0.0%    0.042    0.0%    62.5%   50.0
open_domain        2    50.0%    0.000    0.0%    50.0%   50.0
temporal          10     0.0%    0.000    0.0%    90.0%   50.0
Stage 3 escalation  : 80.4%
mean store size     : 366.0 memories
```

5% is far below the 30–50% the plan predicted, and low enough that the right assumption is
a defect, not weak performance. This is what a full read of the pipeline turned up, ordered
by how much of the gap each item explains.

Two of these are bugs in the benchmark harness (mine). Three are design choices in Memora
that are defensible for its intended use and wrong for QA. One is a latent bug in Memora
that was invisible in every statistic. They are separated below, because only the harness
bugs are unambiguously "fix and re-run" — changing Memora changes the system under test and
has to be reported as such.

---

## The headline: temporal questions were unanswerable by construction

**90% abstention on temporal, 0% correct.** Not a ranking problem. The reader was never
given a single date.

Memories reach the reader through `MemoryRetriever.format_memories_for_prompt`
(`src/retriever.py:426`), which renders each one as:

```
- museum_visit: went to the art museum [turn 143, 85% confident]
```

There is no date anywhere in that line. A LoCoMo temporal question — *"When did Melanie
visit the art museum?"* — cannot be answered from `turn 143`. The reader is instructed to
reply `NO_ANSWER` when the context lacks the answer, and it correctly did so 9 times out
of 10.

It is worse than a missing field, because the obvious fix is also wrong. Memory records
*do* carry a `timestamp` (`MEMORY_FIELDS`, `src/config.py:115`) — but it is set from
`datetime.now()` at ingest (`src/extractor.py:143`, `src/llm_extractor.py:398`). For a
replayed 2023 conversation ingested in 2026, that timestamp is the *benchmark run's* wall
clock. Surfacing it would have produced confidently wrong dates instead of abstentions,
which is a considerably worse failure.

The root cause is an API gap noted as blocker **B2** in `BENCHMARK_PLAN.md`:
`process_turn(user_message)` accepts no timestamp, so the adapter folds the date into the
text (`[8 May, 2023] Melanie: ...`, `benchmarks/dataset.py:90`). Extraction sees it;
retrieval discards it.

**Fix, implemented:** the full turn text is already persisted as `source_text`, so the date
is recoverable without a schema change, without re-extraction, and without re-ingesting
anything. `MEMORY_CONTEXT_INCLUDE_DATE=true` prefixes it:

```
- museum_visit: went to the art museum [8 May, 2023, turn 143, 85% confident]
```

Off by default — it changes the prompt for every caller, and on a live assistant the ingest
date and the event date normally coincide, so this is a benchmark-shaped problem.

Temporal is roughly a fifth of LoCoMo, currently scoring 0%.

---

## Ranking optimises for an assistant, not for question answering

`RANKING_WEIGHTS_5_SIGNAL` (`src/config.py:255`) weights **type at 0.40 — above semantic
at 0.30.** Combined with `TYPE_PRIORITIES` (`src/config.py:85`), that is actively hostile
to QA, because LoCoMo answers live in exactly the two lowest-priority types:

| type | priority | contribution at weight 0.40 |
|---|---|---|
| constraint | 1.00 | **0.400** |
| instruction | 0.95 | 0.380 |
| commitment | 0.80 | 0.320 |
| preference | 0.70 | 0.280 |
| entity | 0.60 | 0.240 |
| **fact** | 0.50 | **0.200** |
| **event** | 0.40 | **0.160** |

A semantically *perfect* match (1.0) contributes 0.30. So a **completely irrelevant
constraint scores 0.40 from type alone and outranks a perfectly matching event** at
0.30 + 0.16 = 0.46 only narrowly — and beats a merely good match (semantic 0.6 →
0.18 + 0.16 = 0.34) outright.

This is a reasonable trade for the product Memora is: a dietary constraint should surface
whether or not the user just mentioned it. It is the wrong trade for answering questions
about what was said. Compounding it, `constraint` and `instruction` are injected
unconditionally as always-on types, consuming top-K slots regardless of relevance.

**Fix, implemented:** weights are now env-overridable (`RANK_W_SEMANTIC`, `RANK_W_TYPE`, …)
with defaults unchanged, so this can be ablated without editing config and the baseline
stays exactly the shipped behaviour.

---

## Retrieval was saturating its own cap

`retr` is **exactly 50.0** for every category — `MAX_MEMORIES_TO_RETRIEVE`, pinned. Against
a mean store of 366, retrieval returns the top 13.7% every time, and the cap binds on every
question. Contributing factors:

- `RECENCY_RETRIEVAL_LIMIT = 50` equals `MAX_MEMORIES_TO_RETRIEVE = 50`, so the recency
  branch alone can fill the entire result set. Those memories bypass the similarity floor
  and are assigned a flat `RECENCY_FALLBACK_SEMANTIC_SCORE = 0.15` regardless of relevance.
- `MIN_SEMANTIC_SCORE = 0.1` is low enough to admit nearly anything.
- The token-budget trim never fires: `MEMORY_TOKEN_BUDGET // 50 = 60`, above the cap of 50.

So the context is 50 memories chosen substantially by type and recency rather than
relevance — consistent with high abstention across *all* categories, not just temporal.

---

## Latent bug: retrieved memories that never reached the prompt

`format_memories_for_prompt` bucketed memories into a fixed 7-key dict and **silently
dropped anything whose type was not one of them**:

```python
if mem_type in sections:
    sections[mem_type].append(mem)
# else: gone
```

Such a memory was still retrieved, still counted in `retrieved_count`, and still had its
`access_count` incremented — it simply never appeared in the context. Every statistic
reported it as delivered.

Stage 3 is *asked* for one of `MEMORY_TYPES`, and mostly complies, so this is unlikely to
be a large share of the 5%. But with **80.4% of turns escalating to the LLM**, the exposure
is far higher than the 13% the design assumed, and the failure was undetectable.

**Fix, implemented:** unrecognised types are bucketed as `fact` and logged at debug, so a
retrieved memory always reaches the prompt.

---

## Harness bugs (mine) that inflated the damage

**1. The sample excluded the easy categories.** `--max-questions` took `questions[:N]`.
LoCoMo groups questions by category, so 20 questions gave 8 multi-hop, 10 temporal, 2
open-domain — **zero single-hop, zero adversarial**. Single-hop is the easiest category;
adversarial rewards abstention, which this system does constantly and would have scored
*well* on. Fixed: round-robin stratified sampling (`cc35118`).

**2. Progress was invisible**, which is why the run looked hung. Fixed in `3c9c946`.

Neither changes what Memora did — but the *reported* 5% is not an estimate of the full
benchmark, because the sample was drawn from the two hardest categories only.

---

## What I did not find

- **No silent Qdrant degradation.** Preflight passed and `report` raised no
  `vector_store_down` flag, so semantic retrieval was live.
- **No judge miscalibration.** The judge handles the category-5 special case, and token-F1
  (0.042 / 0.000) agrees with the judge that the answers were genuinely wrong — this is not
  a grading artefact.
- **No reader-prompt defect.** Given a context with no dates, `NO_ANSWER` is the correct
  behaviour. The reader was right.

---

## Recommended order of work

Run these as ablations against the same conversation. `--reuse-store` makes each one
seconds instead of ~6 minutes, and all of these change only *retrieval and formatting*, so
the ingested store stays valid.

```bash
# 0. Baseline, stratified across all five categories (re-ingest once)
python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context --force
python -m benchmarks.report && python -m benchmarks.diagnose

# 1. Dates in context. Expected to move temporal off 0%.
MEMORY_CONTEXT_INCLUDE_DATE=true \
  python run_locomo.py --limit 1 --max-questions 25 --workers 1 \
                       --save-context --force --reuse-store
python -m benchmarks.report

# 2. Relevance-weighted ranking, on top of dates.
MEMORY_CONTEXT_INCLUDE_DATE=true RANK_W_SEMANTIC=0.55 RANK_W_TYPE=0.15 \
  python run_locomo.py --limit 1 --max-questions 25 --workers 1 \
                       --save-context --force --reuse-store
python -m benchmarks.report
```

Run `benchmarks.diagnose` after each — it splits wrong answers into *retrieval misses* (the
fact never reached the reader) and *reader misses* (it did, and the answer still failed),
which is the only way to tell whether a change helped for the reason you think.

**Expected:** step 1 should be the large one, since it converts a structurally impossible
category into a merely hard one. Step 2 should help multi-hop and open-domain. If the
combined result is still far below 30%, the next suspects are extraction quality at 80%
escalation (are the right facts being stored at all?) and the flat 50-memory context, which
`benchmarks.diagnose` can distinguish.

**Reporting requirement:** any published number must state which of these were enabled.
Both default to off, so the shipped-configuration baseline stays measurable and honest.
