# The conversation profile

A redesign of Memora's memory model for **multi-party, time-anchored conversation** — the
setting every conversational-memory benchmark measures, and the one the original design
does not represent.

Select with `MEMORA_PROFILE`:

| value | meaning |
|---|---|
| `conversation` | **default.** Everything below. |
| `legacy` | byte-identical to the pre-redesign system, for A/B comparison. |

Every mechanism is also individually switchable, and each results file records which were
active (`config.profile` and friends), so no number can be quoted without stating what
produced it.

---

## The mismatch this fixes

Memora was built for **one user talking about themselves in the present tense** — "I prefer
X", "my name is Y", "I'm allergic to Z". Its whole shape follows from that: one implicit
speaker, memories keyed by topic, ranked by how important that *kind* of memory is.

A conversational benchmark is a different problem. Several people recount events that
happened on particular dates, across sessions spanning months, and are then asked who did
what, when, and what followed from it. Four assumptions break — all silently, none
producing an error:

| assumption | consequence |
|---|---|
| one speaker | facts about different people merge; attribution is quietly wrong |
| now == when it happened | "when" questions are unanswerable |
| `(type, key)` is unique | the first `entity:location` blocks every later one, forever |
| type priority ≈ relevance | irrelevant constraints outrank the memory that answers the question |

None of these are bugs in the assistant setting. All of them are fatal outside it.

---

## 1. Speaker and event time as first-class fields

`process_turn` grew three optional parameters:

```python
system.process_turn(text, speaker="Melanie",
                    event_date="8 May, 2023", event_ts=1683504000.0)
```

Three new record fields — `speaker`, `event_date`, `event_ts` — plumbed through all four
places a Memora field has to exist (config, Redis write, Redis read-cast, vector payload).

**Why not keep folding it into the text.** The benchmark adapter used to prefix
`[8 May, 2023] Melanie: ...` and hope extraction preserved it. That makes provenance
*extraction-dependent*: whether a fact keeps its date comes down to whether the LLM chose
to copy it into the value. As real fields they are always present, always indexable, and
rankable.

**Why `event_date` is not `timestamp`.** `timestamp` is ingest wall-clock. For a 2023
conversation replayed in 2026 it is off by three years — so surfacing it would produce
confidently wrong dates, which is worse than abstaining.

## 2. Dedup identity that does not annihilate facts

Legacy identity is `(type, key)`, global and permanent. In a long multi-party conversation
`entity:location` occurs constantly — a second speaker's home, the same speaker moving —
and **every occurrence after the first is discarded**, with only a confidence bump on the
original. Turns are consumed, extraction succeeds, and the memory simply never exists.

Identity is now `(type, key, speaker, sha1(normalised value))`. Verbatim repeats still
collapse, which is all this index was for. "This replaces that" is superseding's job, not
an implicit side effect of a key collision.

Semantic dedup gained a matching guard: two recountings of the same *kind* of event on
*different dates* are two events. Cosine similarity cannot see dates, so it is checked
explicitly — merging them erases exactly the distinctions temporal and multi-hop questions
probe.

## 3. Hybrid retrieval: BM25 + dense, fused by RRF

`src/lexical_index.py` adds Okapi BM25 over the store, fused with dense search by
Reciprocal Rank Fusion.

**Why a second channel rather than better tuning.** A 384-dim MiniLM is good at paraphrase
and consistently weak on **rare proper nouns** — names, places, titles — because a rare
token contributes little to a sentence embedding. Those same tokens are *maximally*
informative in the IDF sense, which is what BM25 rewards. The two methods fail on different
queries, so their union recalls more than the better of them alone. That is an argument
tuning cannot replicate.

**Why RRF.** It fuses by *rank*, not score. A cosine and a BM25 magnitude are not
comparable, and normalising between them needs constants that do not transfer between
corpora. RRF needs none.

BM25 also indexes `source_text`, so the original utterance is searchable even when the
extracted `key: value` dropped the proper noun the question uses.

No new dependency; the index rebuilds when the store size changes.

## 4. Ranking rebalanced for retrieval, not importance

| signal | legacy | conversation |
|---|---|---|
| semantic (fused) | 0.30 | **0.55** |
| type | **0.40** | 0.10 |
| confidence | 0.15 | 0.20 |
| recency | 0.10 | 0.10 |
| frequency | 0.05 | 0.05 |

Type priority answers *"how important is this kind of memory in general"* — the right
question for what an assistant must never forget, the wrong one for which memory answers
the question in front of you. Under legacy weights an irrelevant `constraint` scores 0.40
from type alone and beats a well-matching `event` at 0.34 — and conversational answers live
almost entirely in `fact` and `event`, the two *lowest* priorities.

## 5. Chronological, attributed context

Legacy renders memories grouped by type, discarding order:

```
=== EVENT ===
- museum_visit: went to the art museum [turn 40, 90% confident]
```

The conversation profile renders a timeline:

```
=== TIMELINE (oldest first) ===
- [8 May, 2023] Melanie - museum visit: went to the art museum (event)
    said: "[8 May, 2023] Melanie: I finally went to the Ravensbourne art museum today"
- [12 June, 2023] Caroline - hobby: started guitar lessons (fact)

=== UNDATED ===
- Melanie - coffee: prefers espresso (preference)
```

Sequence carries most of the signal for "what happened when" and for chaining facts across
sessions, and a reader cannot reconstruct it from `turn N` scattered across type sections.
Undated memories keep their own heading rather than corrupting the timeline.

`said:` attaches the originating utterance to the top few memories only
(`CONTEXT_EVIDENCE_TOP_N=8`). `key: value` is a lossy compression; for the best-ranked
memories the raw sentence often holds the exact detail asked for.

## 6. Embedding text that looks like language

| | text embedded |
|---|---|
| legacy | `museum visit \| went to the art museum \| type: event` |
| conversation | `Melanie - museum visit: went to the art museum (on 8 May, 2023)` |

Two problems with the legacy form. Queries are natural-language questions, and sentence
encoders are trained on sentence pairs — comparing a fragment to a sentence puts them in
systematically different regions. And `type: event` appears verbatim in every event memory,
adding a constant component to thousands of vectors that compresses the very distinctions
being searched on.

## 7. Query-aware retrieval

Two small additive boosts, computed from the **shape of the query**:

- **Temporal intent** — "when / what year / how long ago" ⇒ prefer memories carrying a date.
- **Speaker match** — a capitalised name in the query ⇒ prefer that speaker's memories.

Both are generic properties of question form. **No dataset vocabulary appears in either,
and none should.** They are additive and small (0.15) so they reorder near-ties rather than
overriding relevance.

## 8. Multi-hop expansion

A multi-hop question names one entity and asks about another reachable only through it. The
second entity is *not in the query*, so one similarity lookup cannot bridge the gap.

A second pass re-queries with the original question plus the content of the best first-pass
memories. New candidates enter at 0.6× score — reached indirectly, so they should rank below
anything matched directly.

---

## What was deliberately not done

Things that would raise the score and are **cheating**, so they are absent:

- No reading questions, answers, or category labels at ingest time.
- No LoCoMo-specific vocabulary, regexes, date formats, or speaker names in `src/`.
- No per-category branching anywhere in retrieval or the reader.
- No tuning against gold answers.
- No retrieval of the raw conversation as a fallback — memories only.

The dataset adapter in `benchmarks/` knows LoCoMo's *file format*, which is its job. Nothing
in `src/` knows the benchmark exists.

---

## Running it

```bash
# new architecture (default)
python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context --force
python -m benchmarks.report && python -m benchmarks.diagnose

# baseline for comparison
MEMORA_PROFILE=legacy python run_locomo.py --limit 1 --max-questions 25 \
    --workers 1 --save-context --force

# isolate one mechanism
LEXICAL_SEARCH_ENABLED=false python run_locomo.py ...
```

**Re-ingest is required** when changing anything that affects storage — the dedup identity,
provenance fields, and embedding text all change what gets written. `--reuse-store` is only
valid for retrieval- and rendering-side changes (context, ranking weights, query-aware,
multi-hop).

## Expected effect

Ordered by how much of the current 5% each should recover:

1. **Dedup identity** — the largest, and the least visible. Everything downstream is
   capped by what actually got stored.
2. **Dates in context** — converts temporal from structurally impossible to merely hard.
3. **BM25 fusion** — proper-noun questions across every category.
4. **Ranking weights** — stops irrelevant constraints occupying the context.
5. **Chronological rendering + speaker** — multi-hop and temporal reasoning.
6. **Multi-hop expansion, query-aware boosts** — narrower, smaller.

Measure them separately. If the combined result is still far below 30%, the next suspect is
extraction quality at ~80% Stage 3 escalation — whether the right facts are being stored at
all — which `benchmarks.diagnose` distinguishes from ranking failure.
