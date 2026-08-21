# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Memora is a long-form memory system for AI agents (Python library, no server/API layer). `MemorySystem.process_turn(user_message)` runs the full pipeline — extract → store → dedup → retrieve → format — and returns `(memory_context_string, stats_dict)`. The caller is responsible for actually calling an LLM with the returned context; the library never generates responses.

Since mid-2026 the project has a second half: **`benchmarks/`, a LoCoMo evaluation harness**. Recent architectural change has been driven by measured benchmark failures rather than by feature work. If you are asked to "improve recall" or "fix the score", the harness is how you find out whether you did — see *Benchmarking* below.

The codebase is organized as "Phases 1–5" (that vocabulary is everywhere in comments, configs, and docs). Phases are not directories — they're feature layers stacked in the same files:

| Phase | Adds |
|---|---|
| 1 | Flat-file core memory, Redis long-term store, heuristic + regex extraction, recency/type retrieval |
| 2 | Qdrant vector store, sentence-transformers embeddings, 3-signal ranking |
| 3 | LLM (Stage 3) extraction, semantic dedup, superseding, confidence modifiers |
| 4 | Consolidation worker (decay/merge/promote), 5-signal ranking, access tracking |
| 5 | `evaluation/` RAGAS-style harness |
| 6 (unnamed) | `benchmarks/` LoCoMo harness, BM25 lexical index, entity index, profiles |

## Commands

Requires Redis + Qdrant running. `.venv/` exists in the main checkout (gitignored, so **not** present in worktrees); activate it (`.venv\Scripts\Activate.ps1`) or use `.venv/Scripts/python.exe`. On the Linux server the venv is `.venv-linux/`.

```powershell
docker-compose up -d             # Redis :6379 + Qdrant :6333 (make start / make stop also work)
bash scripts/start_backends.sh   # docker-or-native fallback; use when Docker is unavailable
pip install -r requirements.txt
python demo_phase4.py                    # consolidation + 5-signal demo, checks deps first
python demo_active_memories.py           # 10 turns, shows active_memories tracking
python example_json_logging.py           # per-turn JSON logging
python test_customer_conversation.py     # 60-turn customer-service scenario
python test_comprehensive_1000_turn.py   # 1000 turns, all phases (slow, burns API quota)
```

**`python -m benchmarks.selftest` is the fast check and the closest thing to a unit suite** — 141 logic assertions covering config invariants, prompt contents, ranking, dedup keys, the entity index, extraction-cache keying and failure handling, cross-encoder reranking, and the results reader/writer contract. It needs no Redis, no Qdrant, and no network, so it runs anywhere in seconds. Run it before and after any change to `src/` or `benchmarks/`. It is the only test that catches a config or prompt regression before you spend an hour of compute on a benchmark run.

There is **no pytest suite** — `pytest` is listed in `requirements_evaluation.txt` but no test functions are collected, and `make test` is a stub.

Evaluation (Phase 5) needs the heavier extra deps:

```powershell
pip install -r requirements_evaluation.txt
python run_evaluation.py      # generates fixtures if absent, health-checks, runs full suite
```

`evaluation/fixtures/*.json` are cached — `run_evaluation.py` regenerates them only when `test_conversations.json` is missing. Delete it to force fresh data (`ConversationGenerator(seed=42)`).

## Benchmarking (LoCoMo)

LoCoMo is 10 conversations / 5,882 turns / 1,986 gradable questions across 5 categories (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial). Primary metric is LLM-as-judge; token-F1 and exact-match are kept as deterministic secondaries.

```bash
python -m benchmarks.preflight                  # verify backends, deps, keys are live
python -m benchmarks.estimate --workers 8       # cost/time projection before committing
python run_locomo.py --limit 1 --max-questions 25 --workers 1 --save-context   # SMOKE TEST FIRST
python -m benchmarks.report                     # scorecard
python -m benchmarks.diagnose                   # WHY it scored that — see below
```

Runs are **resumable**: conversations already in `results/locomo/raw/` are skipped. `--force` redoes them.

**`--reuse-store` skips re-ingestion.** It is only valid when your change affects *retrieval or reading*. Any change to extraction, storage, dedup keys, or the entity index requires a **full re-ingest**, because those run at write time — reusing a store silently measures the old pipeline and produces a confidently wrong A/B.

**`benchmarks/diagnose.py` matters more than the score.** It splits failures into *retrieval misses* (gold words never reached the context — extraction or ranking is at fault) and *reader misses* (the answer was present and the reader still failed). Those two have completely different fixes, and the headline percentage cannot distinguish them. It also reports a **floor** score by auditing abstentions: LoCoMo's "adversarial" golds often contain real facts, so a refusal there is usually a genuine failure rather than a judge artifact. Quote the floor, not the ceiling.

**Small samples lie.** Successive `n=25` single-conversation readings have swung 64% → 32% → 48% on changes that were partly confounded. Anything below a full 10-conversation run is a smoke test, not a result.

**Confounding to watch:** `STAGE_3_TEMPERATURE` defaults to `0.0` precisely so that re-ingesting builds the *same* store twice. Setting it non-zero makes every A/B compare two different memory stores, and the retrieval delta becomes unreadable.

**Temperature 0 is not enough on a reasoning model.** `gpt-oss-120b` spends hidden chain-of-thought before its visible reply and that is not bit-exact run to run: the same conversation re-ingested twice produced 769 memories and then 431, with identical turn count, escalation rate and empty rate. The **Stage 3 extraction cache** (`src/extraction_cache.py`) fixes this by memoising extraction on disk, keyed by a hash of the fully-assembled prompt plus provider/model/temperature/max-tokens/turn. It is **on by default for benchmark runs** (`--no-extraction-cache` to disable) and off for the library.

Two consequences worth internalising:

- Editing the prompt, the VALUE QUALITY block, the model or the temperature changes the key automatically, so a stale cache cannot serve results from a pipeline that no longer exists. Changing the *parse/validate output shape* is the one thing the hash cannot see — bump `EXTRACTION_CACHE_VERSION`.
- A re-ingest after the first one is free and bit-identical, so **prefer a full re-ingest over `--reuse-store`**. Rebuilding exactly is now cheaper than reasoning about whether reuse is valid.

Failures are never cached: `parse_error`, `invalid_schema` and `api_error` are recorded as reasons and skipped, so a transient failure cannot be frozen and replayed as "nothing worth remembering". The breakdown appears as `ingest.stage3_empty_reasons` — check it before believing a low extraction count.

**Retrieval ablations that are wired and unswept:** `--top-k` (the cap binds on *every* question at its default 50) and `--rerank` (cross-encoder over the top 50 candidates, applied before the top-K cut so it can promote a memory ranking placed outside it). Both are retrieval-only, so `--reuse-store` is valid; both are off by default and recorded in each results file's `config` block.

## Architecture

### Profiles: `MEMORA_PROFILE`

`src/config.py` branches on `MEMORA_PROFILE` (`conversation`, the default, vs `legacy`). This exists so benchmark-driven retuning could happen without silently changing behaviour for the original assistant-style use case. `conversation` enables the lexical index, the entity index, speaker/value-aware dedup keys, a low always-on floor, and relevance-weighted ranking; `legacy` restores the pre-benchmark values.

Consequences when working here:

- **Config freezes env at import.** Changing `MEMORA_PROFILE` (or any `MEMORA_*` env var) at runtime does nothing unless you reload `src.*` — this is why `benchmarks/selftest.py` has a `_reload(profile)` helper. Tests asserting on both profiles must use it.
- Ranking weights differ by profile: conversation is `semantic .55 / type .10 / recency .10 / frequency .05 / confidence .20`; legacy is `semantic .30 / type .40 / ...`. The high `type` weight in legacy means an irrelevant `constraint` outranks a well-matching `event` — deliberate for instruction-following, wrong for question answering.

### Storage: four layers, authoritative for different things

- **Core memory** — `memory/<user_id>/{CORE,PREFERENCES,INSTRUCTIONS,CONSTRAINTS}.md`, human-editable Markdown, **always** injected verbatim (`FlatFileStore.read_core_memory`). Templates are created on first `FlatFileStore.__init__`; the `##` section headers in those templates are the promotion targets, so changing a header silently breaks Phase 4 promotion (`consolidation_worker.py:491-494` maps type → file+section).
- **Long-term memory (authoritative)** — Redis hashes at `mem:<id>`, plus indices: `type:<type>` (set), `dedup:<type>:<key>` (string), `recent_memories` (sorted set by timestamp; also the source of truth for `count_memories`), and `ent:<entity>` (set, the entity index).
- **Vector memory (search only)** — Qdrant collection `memory_vectors`, 384-dim cosine (`all-MiniLM-L6-v2`). Payload is not authoritative: retrieval hydrates each hit from Redis and falls back to the payload only if Redis lost it.
- **Lexical index (search only)** — `src/lexical_index.py`, in-process BM25 over memory text. No new dependency.

### Pipeline (`memory_system.py:process_turn`)

Extraction is a 3-stage cascade in `extractor.py` — each stage only runs if the previous was inconclusive:

1. **Stage 1 heuristic filter** (`should_extract`) — weighted length/keyword/question/specificity score vs `SENSORY_FILTER_THRESHOLD`. Rejects greetings and sub-5-char messages outright.
2. **Stage 2 regex classifier** (`classify_and_extract`) — pattern tables per memory type, plus a payment/financial block. Note Stage 2 matches most patterns against `message.lower()` but the payment patterns against raw `message` (they depend on capitalized dates/names).
3. **Stage 3 LLM** (`llm_extractor.py`) — escalated when Stage 2 returned nothing (and heuristic score > 0.5) or its best confidence < `STAGE_3_CONFIDENCE_THRESHOLD`. Results merged by `(type, key)`, higher confidence wins. Groq/OpenAI/Anthropic behind lazy per-provider clients; Groq supports N-key rotation on 429 (`GROQ_API_KEYS`, clients built with `max_retries=0` so rotation, not the SDK, handles retries).

On dense conversational data ~80% of turns reach Stage 3, which is the dominant benchmark cost. Stages 1–2 were tuned for assistant-style input and generalize poorly to narrative dialogue.

`_enrich_temporal()` runs **after** the escalation decision, recovering dates the LLM dropped from the value without suppressing escalation. Ordering matters — enriching first would make turns look already-handled.

Then: semantic dedup (cosine ≥ `SEMANTIC_DEDUP_THRESHOLD` within same type+user) either boosts the existing memory's confidence or, if `is_update` is set, stores the new one and marks the old `superseded_by`.

Retrieval (`retriever.py:_retrieve_with_semantic_search`) is a **multi-branch hybrid**, fused and then re-scored:

- semantic (Qdrant) branch
- lexical BM25 branch, fused with the dense branch via **Reciprocal Rank Fusion** (`RRF_K=60`) — RRF fuses by *rank*, avoiding the cross-corpus score-normalization constants that would otherwise need tuning
- **entity branch** (`src/entity_index.py`) — retrieves memories mentioning the query's entities, scored `min(1.0, ENTITY_MATCH_SCORE * hits)` where `hits` counts how many *distinct* query entities a memory mentions, so a memory naming two queried people outranks one naming either. This is the cheap substitute for a knowledge graph: no schema, no entity resolution, and it recovers subject-relevant memories that share almost no wording with the question. It cannot chain inferences across relations — that remains the honest argument for a real KG later.
- recency branch that deliberately bypasses the similarity floor (assigned `RECENCY_FALLBACK_SEMANTIC_SCORE`) — this is what gets long-distance recall
- always-on `constraint`/`instruction` types

then a 5-signal weighted sum, superseded-filter, optional cross-encoder rerank (`src/reranker.py`, off by default — runs *before* the cut so it can promote a memory the sum ranked outside top-K), top-K, token-budget trim. Note the token-budget trim cannot fire at the shipped defaults: `MEMORY_TOKEN_BUDGET // TOKENS_PER_MEMORY_ESTIMATE` is 60, above `MAX_MEMORIES_TO_RETRIEVE` of 50, so top-K alone bounds context size.

Consolidation runs inline, not in a thread: `process_turn` calls `ConsolidationWorker.needs_consolidation` and blocks on `run_consolidation` every `CONSOLIDATION_INTERVAL_TURNS`. Decay, merge, and promote are each independently gated by their own flag (`MEMORY_DECAY_ENABLED` is currently **False**).

### Optional-dependency degradation

`src/__init__.py`, `memory_system.py`, and `extractor.py` all lazy-import their optional layers inside `try/except` and log a warning on failure. Qdrant down or `sentence-transformers` missing ⇒ the system silently drops to Phase 1 retrieval; Groq key missing ⇒ Stage 3 disabled. So a "recall regression" is often just a backend that failed to connect — check `MemorySystem.health_check()` and the startup warnings before touching ranking weights.

## Conventions and traps

**`src/config.py` is the single source of truth for all tuning.** Nothing is hardcoded at call sites; every threshold, weight, and prefix is imported from it. The prose docs (`QUICK_REFERENCE.md`, `README.md`, `RESULTS_FEBRUARY_2026.md`) have drifted from it — e.g. they claim `MIN_SEMANTIC_SCORE = 0.3`, `MAX_MEMORIES_TO_RETRIEVE = 10`, `MEMORY_TOKEN_BUDGET = 500`, and a `stats['memories_extracted']` key, none of which match the code (`0.1`, `50`, `3000`, `stats['extracted_count']`). Read `config.py`, not the docs, and don't "fix" code to match a doc.

**Blanket excepts hide extraction bugs as "nothing worth remembering".** `LLMExtractor.extract()` wraps its body in a bare `except` that returns `[]`. A truncated JSON response, a missing attribute, or a raised `RecursionError` all present identically to a turn that genuinely had no facts — silently, at ingest time, across thousands of turns. When extraction "finds nothing", verify against a single turn with logging before believing it. (`STAGE_3_MAX_TOKENS` was raised 500 → 1200 for exactly this reason: richer values overflowed the cap and the truncated JSON was swallowed.)

**The JSON retry is capped at one attempt on purpose.** `_parse_and_validate` → `_retry_with_error` → `_call_llm` → `_parse_and_validate` is a cycle; the `attempt` parameter is what bounds it. Without that guard, a model that never emits JSON recurses until `RecursionError` — one API call per stack frame. Don't drop it when refactoring.

**Extraction quality is bounded by the prompt's examples, not its rules.** The base `EXTRACTION_PROMPT`'s examples all have bare-token values (`"Alex"`, `"Google"`), which teaches maximal compression — producing memories like `charity race: 18 May 2023` that drop the very purpose a question asks about. The conversation preamble in `llm_extractor.py` counteracts this with an explicit VALUE QUALITY block and a labelled counter-example. **No ranking change recovers a detail extraction discarded**, so extraction fixes come before retrieval fixes.

**Redis is not user-namespaced.** Keys are global (`mem:`, `type:`, `dedup:`, `recent_memories`, `ent:`) — only flat files (per-user directory) and Qdrant (`user_id` payload filter) isolate users. Consequences: two `MemorySystem` instances with different `user_id`s share the same Redis long-term store; `count_memories()` is global; `clear_memories()` wipes every user's memories **and** deletes the entire Qdrant collection. The evaluator and the LoCoMo runner both call `clear_memories()` per conversation, so never point either at a Redis/Qdrant instance holding data you care about.

**Dedup keys must include speaker and value.** `build_dedup_key()` composes `type + key [+ speaker] [+ sha1(value)[:16]]` under the conversation profile. With the bare `type:key` form, two speakers' distinct facts sharing a key collided globally and one was silently dropped — fixing this roughly doubled store size (366 → 868 memories) on the same input.

**Adding a memory field is a four-place change.** Redis hashes can't hold `None` or `bool`, so the write path coerces (`None → ''`, `bool → str`) and `RedisStore.get_memory` re-casts field by field. A new field needs: `MEMORY_FIELDS` in config, a `setdefault` in `store_memory`, a cast in `get_memory`, and — if it should be searchable — the payload dict in `VectorStore.store_memory`. Skip the cast and downstream code gets a string where it expects a number. (`speaker`, `event_date`, `event_ts` were added this way; `event_ts` needs the float cast.)

**`MemorySystem.get_prompt_context` returns a tuple**, not the `str` its annotation and docstring promise — it forwards `_compose_prompt_context`'s `(context, active_memories)`. Callers in the demos unpack accordingly.

**Access counts are incremented twice per retrieval** — once in `retriever._retrieve_with_semantic_search` and again in `memory_system._compose_prompt_context`. It inflates the frequency signal (weighted 0.05, so effects are small). Fix the duplication rather than compensating in the weights.

**Prompt position beats prompt instruction.** A persistent over-abstention bug survived three rewrites of the reader *system* prompt; the actual cause was the trailing `(or NO_ANSWER)` on the last line of the *user* message. The final line of the user turn outweighs system-prompt guidance — check it first when a model won't stop doing something.

**Repo hygiene:** `.gitignore` excludes `test_*.py` (whitelisting only `test_customer_conversation.py`), `memory/*/`, `output/`, `results/`, and `*.log` — so `test_comprehensive_1000_turn.py` and per-user memory dirs are untracked by design. The README and QUICK_REFERENCE also reference scripts that don't exist (`demo.py` via `make demo`, `test_all_phases.py`, `test_1000_turn_latency.py`, `diagnostic_extraction_phases.py`); use the scripts listed under Commands above instead.

## Server notes (Linux benchmark host)

The benchmark host keeps **everything** under `/home/kenton/projects/memora` — dataset, model cache, Redis/Qdrant volumes, results, logs. Preserve that; nothing should write to `/tmp`, `~/.cache`, or system paths.

- `scripts/start_backends.sh` falls back to native binaries when Docker is unavailable or the daemon is unreachable. It prefers **musl** Qdrant builds — the gnu builds require GLIBC 2.38, newer than the host's.
- systemd-logind `KillUserProcesses=yes` kills user processes at logout, which repeatedly took the backends down mid-run. The script uses `setsid` to detach; long runs should be started the same way.
- If `preflight` reports Python 3.10 rather than 3.12, the venv is not activated.
