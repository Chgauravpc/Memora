# Why the first LoCoMo run scored 5%

> **Status — 2026-08-20.** Everything in *Recommended order of work* below has been
> executed, plus several items it did not anticipate. **Jump to
> [Status after the fixes](#status-after-the-fixes) for the 2026-08-18 state, then
> [Status log (2026-08-20 on)](#status-log-2026-08-20-on) for every run since** — the model
> Groq decommissioned mid-project, the fixes that followed, and the readings so far, each
> entry appended rather than overwritten. The analysis below is retained as the record of
> the original diagnosis.

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

---

# Status after the fixes

*Last updated 2026-08-18, through commit `fd65ca3`.*

The ablation ladder above was run and then overtaken: the first two steps landed, and the
diagnosis they produced redirected the work toward extraction and reproducibility. What
follows is the current state, so a new session does not re-derive it.

## Score trajectory (all conversation 1, n=25 — smoke tests, not results)

| Reading | Reported | Floor | Note |
|---|---|---|---|
| Initial | 5.0% | — | Substantially a harness artefact: head-sliced sample excluded single-hop and adversarial |
| After dates + ranking + harness fixes | 64.0% | 56.0% | Failures dominated by *reader* misses |
| Next | 32.0% | 28.0% | **Confounded** — `STAGE_3_TEMPERATURE=0.1` rebuilt a different store (coverage 65% → 42%) |
| After pinning temperature to 0 | 48.0% | 40.0% | Failures dominated by *retrieval* misses (8 of 13) |

**None of these is a result.** They are single-conversation, 25-question smoke tests, and
the swing between them is larger than most of the effects being measured. The only number
worth publishing comes from a full 10-conversation run, quoted as the floor.

## What changed, and why

- **Reproducibility first** (`2f98361`). `STAGE_3_TEMPERATURE` → 0.0. Until this landed,
  every re-ingest built a different store and no A/B was readable. The 64% → 32% "regression"
  was this and nothing else.
- **Reader over-abstention** (`f36ddfb`, `0c9d23d`). Three system-prompt rewrites failed to
  fix it; the cause was the trailing `(or NO_ANSWER)` on the *last line of the user message*.
  Position beat instruction.
- **Harness bug** (`02d9352`). `diagnose.py` read a `"questions"` key the worker never
  writes (it writes `"records"`), reporting an empty run as a clean one. Now it accepts both
  and raises on an unrecognised payload rather than reporting nothing.
- **Dedup collapse.** `build_dedup_key()` was `type:key`, global across speakers, so distinct
  facts collided and one was dropped. Adding speaker + value hash took the store from 366 to
  868 memories on identical input.
- **Lexical retrieval** (`2f294e3`). BM25 (`src/lexical_index.py`) fused with dense via RRF.
  `BM25_K1` is set to 50 by request, which effectively disables term-frequency saturation —
  worth re-sweeping against the default 1.2 when there is time.
- **Temporal recovery** (`214785b`). `_enrich_temporal()` restores dates the LLM drops,
  ordered *after* the escalation decision so it does not mask turns that still need Stage 3.
- **Extraction over-compression** (`fd65ca3`). The base prompt's bare-token examples
  (`"Alex"`, `"Google"`) taught maximal compression, yielding memories like
  `charity race: 18 May 2023` with the purpose discarded. Countered with an explicit VALUE
  QUALITY block and a labelled counter-example; `STAGE_3_MAX_TOKENS` 500 → 1200.
- **Entity-centric retrieval** (`fd65ca3`). `src/entity_index.py`, scored by how many
  distinct query entities a memory mentions. Chosen over a real knowledge graph: most of the
  multi-hop benefit, none of the schema or entity-resolution cost.
- **Infrastructure** (`f017b4e`). Native-binary fallback for Redis/Qdrant, musl builds,
  `setsid` against logind's `KillUserProcesses`.

## Open — in the order I would take them

1. **A full 10-conversation run.** Every number above is n=25 on one conversation. This is
   the single highest-value remaining action and it is measurement, not code.
2. **Retrieval misses, currently 8 of 13 failures.** The extraction and entity changes in
   `fd65ca3` target these directly but have not yet been measured. If they do not move, the
   ceiling is not extraction verbosity, and the question becomes whether the right facts are
   being *selected* for extraction at all.
3. **Stage 1/2 escalation rate (~80%).** The heuristic and regex layers were tuned for
   assistant-style input and mostly abstain on narrative dialogue, so nearly every turn pays
   for an LLM call. Improving them is a cost and latency win, not an accuracy one.
4. **`BM25_K1` sweep** — 50 vs the 1.2 default, once something else is not moving.
5. **Knowledge graph.** Still the honest answer for inference chains the entity index cannot
   follow (*"2019 breakup + no current partner ⇒ single"*). Deferred deliberately: a KG
   inherits whatever extraction loses, so it is only worth building on top of extraction that
   has been measured good.

## Standing constraints

- **No hardcoding or benchmark-specific cheating.** Every change must be defensible as a
  general improvement to the memory system. `MEMORA_PROFILE` exists so retuning is explicit
  and reversible rather than smuggled into defaults.
- **The entire server operation stays in `/home/kenton/projects/memora`.**
- Quote the **floor**, state the sample size, and say which flags were on.
- **From 2026-08-20 on: every benchmark run gets an entry below** — what changed since the
  last entry, the exact command, and the result (score, floor, store size, diagnose
  breakdown). Existing entries are never rewritten; a new run appends a new dated entry so
  the trajectory stays legible instead of collapsing into "current status" that silently
  loses what was already tried. If a change is later found to be confounded or wrong, say so
  in a later entry — do not edit the old one away.

---

# Status log (2026-08-20 on)

## 2026-08-18/19 — `llama-3.3-70b-versatile` decommissioned by Groq

Not a code change: Groq removed the model this whole document's baseline (`48%`/`40%
floor`, and everything in "Status after the fixes" above) was measured against. Every
extraction and reader/judge call started returning `groq.NotFoundError: model_not_found`,
silently swallowed by `LLMExtractor.extract()`'s blanket except into `[]`, which produced an
empty store and a **0% LoCoMo score** with no direct error surfaced in the scorecard.

**Fix:** moved `LLM_EXTRACTION_MODEL` → `openai/gpt-oss-120b`, `BENCH_LLM_MODEL` →
`openai/gpt-oss-20b` (worktree commits `08beb20`, `20c99ab` — the second one because the
first fix missed `.env.benchmark.example`, whose own `cp .env.benchmark.example .env`
instruction silently re-clobbered a correctly-edited `.env` back to the dead model on the
benchmark host — that round-trip is why the failure recurred after appearing fixed once).

**New confound this introduces, not yet fully resolved:** both replacement models are
*reasoning* models — they spend tokens on hidden chain-of-thought before the visible reply.
Confirmed directly: a "reply with the single word ready" smoke call returned `''` at
`max_tokens=8` and only succeeded at `max_tokens=200` (37 tokens spent to say one word).
Every hardcoded `max_tokens` in the harness (judge: 8, reader: 128, preflight's own smoke
test: 8) was sized for the old non-reasoning model and would silently truncate to empty on
the new ones — commit `e9ff8cc` raised these (judge → 200, reader → 500 → 800, preflight →
64) and added instrumentation (`Answer.empty`, surfaced in `report`/`diagnose`) so a future
truncation shows up as its own stat instead of masquerading as a wrong answer.

`benchmarks/free_tier.py`'s `TIERS` table was still quoting the dead models' limits;
replaced with live-fetched numbers (`08beb20`... completed in a follow-up edit): all three
current candidates (`gpt-oss-120b`, `gpt-oss-20b`, `qwen/qwen3.6-27b`) share identical
free-tier quota (200K TPD / 8K TPM / 1K RPD per account) — unlike the old 70B-vs-8B split,
model choice no longer changes account math, only cost/latency per call.

## 2026-08-19/20 — P0/P1/P2 reader- and extraction-side fixes

Diagnosed from a `--limit 1 --max-questions 25 --workers 1 --save-context --force` smoke
test on the new models (see reading below): failures had flipped from
retrieval-miss-dominated (the pre-decommission baseline: 8 of 13) to **reader-miss-dominated
(9 of 12, 75%)** — extraction/retrieval was working, the answer step was not. Three fixes,
each independently committed and selftest-verified:

- **P0** (`e9ff8cc`) — reader `max_tokens` 500 → 800; added `Answer.empty` (see above).
- **P1** (`8ff901a`) — `_format_chronological` (`src/retriever.py`) already computed the
  top-`CONTEXT_EVIDENCE_TOP_N`-ranked memories and quoted their `source_text`, but rendered
  them in-place inside a up-to-50-item chronological wall. Added a `=== MOST RELEVANT ===`
  section repeating the same top-ranked memories, in relevance order, *before* the timeline —
  addresses "lost in the middle": a real-but-wrong fact winning over a real-and-right one that
  was simply positioned worse. The timeline is unchanged below it. Verified against a
  synthetic reproduction of the charity-race miss (gold "self-care is important" now renders
  first instead of third).
- **P2** (`5196e51`) — `_TEMPORAL_PATTERNS` (`src/extractor.py`) caught absolute dates and "N
  days ago" but nothing for "yesterday" / "the day before" / "last night" — hypothesized as
  the cause of a temporal miss (gold 7 May, predicted 8 May, memory dated by utterance day).
  Added a selftest case for the hypothesis (passes), but **the actual failing question was
  re-tested after this landed and still failed identically** (see below) — the hypothesis was
  not confirmed against the real turn text before shipping the fix, which is the mistake to
  avoid next time: check the raw dataset text for the specific failing case, don't infer the
  mechanism from the gold/predicted pair alone.

### Reading 1 — after the model migration, before P0/P1/P2

`python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context --force`
(conversation 1, `openai/gpt-oss-120b` extraction / `openai/gpt-oss-20b` reader+judge)

| | |
|---|---|
| Score / floor | **54.2% / 54.2%** (no refusal-credit inflation) |
| Store | 769 memories, 419 turns, 80.4% Stage-3 escalation, 17.5% Stage-3-empty |
| Diagnose | 3 retrieval-miss, 9 reader-miss (75%) — dominance flipped from the pre-decommission baseline |

### Reading 2 — after P0 + P1 + P2, full re-ingest

Same command, same conversation, after commits `e9ff8cc`/`8ff901a`/`5196e51`.

| | |
|---|---|
| Score / floor | **33.3% / 33.3%** |
| Store | **431 memories** — same 419 turns, same 80.4% escalation, similar ~19% Stage-3-empty |
| Diagnose | 7 retrieval-miss, 10 reader-miss, mixed |

**This is not read as "P0/P1/P2 made things worse."** None of the three has a code path that
reduces store size: P0 and P1 never run during ingest (reader/judge-time and
retrieval-formatting-time respectively), and P2 only *appends* text to a value, which changes
a dedup hash and should if anything produce *more* distinct memories, not 44% fewer. Turn
count, escalation rate, and empty-call rate were all within noise of Reading 1 — only the
store composition changed. Leading hypothesis: **`openai/gpt-oss-120b`'s Stage-3 output is
not reproducible at `STAGE_3_TEMPERATURE=0.0` the way the old non-reasoning model was** — a
reasoning model's hidden chain-of-thought is not guaranteed bit-exact run to run, and that
was never verified after the model swap, only assumed to carry over. This is the same shape
of confound as the documented 64%→32% swing above, with a different root cause.

**Not yet run: the isolating experiment** — re-run the identical command with zero code
changes and compare store size / score against Reading 2. If it swings again, that confirms
model non-determinism as the dominant noise source and the single-conversation methodology
needs either multiple averaged ingests or the full 10-conversation run before any P0/P1/P2
verdict — good or bad — is trustworthy. **This is now the top open item, above the full
10-conversation run itself**, since a full run inherits the same confound if it's real.

Two specific failures worth carrying forward regardless of the confound, because they
recurred identically across both readings (i.e. survived a full store rebuild):
- Charity-race adversarial question: gold fact not in the `MOST RELEVANT` section in either
  reading, meaning P1 had nothing to promote — the wrong fact outranks the right one, which is
  a ranking problem P1 cannot fix by construction.
- Support-group temporal question: still 8 May vs gold 7 May after P2. The relative-day
  hypothesis is unconfirmed for this specific case — next step is reading the actual raw
  LoCoMo turn text, not re-guessing.
