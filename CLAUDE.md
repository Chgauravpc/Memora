# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Memora is a long-form memory system for AI agents (Python library, no server/API layer). `MemorySystem.process_turn(user_message)` runs the full pipeline — extract → store → dedup → retrieve → format — and returns `(memory_context_string, stats_dict)`. The caller is responsible for actually calling an LLM with the returned context; the library never generates responses.

The codebase is organized as "Phases 1–5" (that vocabulary is everywhere in comments, configs, and docs). Phases are not directories — they're feature layers stacked in the same files:

| Phase | Adds |
|---|---|
| 1 | Flat-file core memory, Redis long-term store, heuristic + regex extraction, recency/type retrieval |
| 2 | Qdrant vector store, sentence-transformers embeddings, 3-signal ranking |
| 3 | LLM (Stage 3) extraction, semantic dedup, superseding, confidence modifiers |
| 4 | Consolidation worker (decay/merge/promote), 5-signal ranking, access tracking |
| 5 | `evaluation/` RAGAS-style harness |

## Commands

Requires Redis + Qdrant running. `.venv/` exists in-repo; activate it (`.venv\Scripts\Activate.ps1`) or use `.venv/Scripts/python.exe`.

```powershell
docker-compose up -d          # Redis :6379 + Qdrant :6333 (make start / make stop also work)
pip install -r requirements.txt
python demo_phase4.py                    # consolidation + 5-signal demo, checks deps first
python demo_active_memories.py           # 10 turns, shows active_memories tracking
python example_json_logging.py           # per-turn JSON logging
python test_customer_conversation.py     # 60-turn customer-service scenario
python test_comprehensive_1000_turn.py   # 1000 turns, all phases, ~3 consolidation cycles (slow, burns API quota)
```

Evaluation (Phase 5) needs the heavier extra deps:

```powershell
pip install -r requirements_evaluation.txt
python run_evaluation.py      # generates fixtures if absent, health-checks, runs full suite
```

There is **no pytest suite** — `pytest` is listed in `requirements_evaluation.txt` but no test functions are collected, and `make test` is a stub. "Running a single test" means running one of the scripts above, or driving one evaluation slice directly:

```python
from evaluation.evaluator import MemorySystemEvaluator
MemorySystemEvaluator().evaluate_retrieval(conversations, verbose=True)   # or evaluate_extraction / distance sweep
```

`evaluation/fixtures/*.json` are cached — `run_evaluation.py` regenerates them only when `test_conversations.json` is missing. Delete it to force fresh data (`ConversationGenerator(seed=42)`).

## Architecture

### Four storage layers, three backends

- **Core memory** — `memory/<user_id>/{CORE,PREFERENCES,INSTRUCTIONS,CONSTRAINTS}.md`, human-editable Markdown, **always** injected verbatim (`FlatFileStore.read_core_memory`). Templates are created on first `FlatFileStore.__init__`; the `##` section headers in those templates are the promotion targets, so changing a header silently breaks Phase 4 promotion (`consolidation_worker.py:491-494` maps type → file+section).
- **Long-term memory** — Redis hashes at `mem:<id>`, plus three indices: `type:<type>` (set), `dedup:<type>:<key>` (string, exact-key dedup), `recent_memories` (sorted set by timestamp; also the source of truth for `count_memories`).
- **Vector memory** — Qdrant collection `memory_vectors`, 384-dim cosine (`all-MiniLM-L6-v2`). Payload only, not authoritative: retrieval hydrates each hit from Redis and falls back to the payload only if Redis lost it.

### Pipeline (`memory_system.py:process_turn`)

Extraction is a 3-stage cascade in `extractor.py` — each stage only runs if the previous was inconclusive, which is why ~87% of turns never hit the LLM:

1. **Stage 1 heuristic filter** (`should_extract`) — weighted length/keyword/question/specificity score vs `SENSORY_FILTER_THRESHOLD`. Rejects greetings and sub-5-char messages outright.
2. **Stage 2 regex classifier** (`classify_and_extract`) — pattern tables per memory type, plus a payment/financial block. Note Stage 2 matches most patterns against `message.lower()` but the payment patterns against raw `message` (they depend on capitalized dates/names).
3. **Stage 3 LLM** (`llm_extractor.py`) — escalated only when Stage 2 returned nothing (and heuristic score > 0.5) or its best confidence < `STAGE_3_CONFIDENCE_THRESHOLD`. Results merged by `(type, key)`, higher confidence wins. Groq/OpenAI/Anthropic behind lazy per-provider clients; Groq supports N-key rotation on 429 (`GROQ_API_KEYS`, clients built with `max_retries=0` so rotation, not the SDK, handles retries).

Then: semantic dedup (cosine ≥ `SEMANTIC_DEDUP_THRESHOLD` within same type+user) either boosts the existing memory's confidence or, if `is_update` is set, stores the new one and marks the old `superseded_by`. Retrieval is **hybrid** (`retriever.py:_retrieve_with_semantic_search`): a semantic branch, a recency branch that deliberately bypasses the similarity floor (assigned `RECENCY_FALLBACK_SEMANTIC_SCORE` — this is what gets long-distance recall), always-on `constraint`/`instruction` types, then a 5-signal weighted sum, superseded-filter, top-K, token-budget trim.

Consolidation runs inline, not in a thread: `process_turn` calls `ConsolidationWorker.needs_consolidation` and blocks on `run_consolidation` every `CONSOLIDATION_INTERVAL_TURNS`. Decay, merge, and promote are each independently gated by their own flag (`MEMORY_DECAY_ENABLED` is currently **False**).

### Optional-dependency degradation

`src/__init__.py`, `memory_system.py`, and `extractor.py` all lazy-import their optional layers inside `try/except` and log a warning on failure. Qdrant down or `sentence-transformers` missing ⇒ the system silently drops to Phase 1 retrieval; Groq key missing ⇒ Stage 3 disabled. So a "recall regression" is often just a backend that failed to connect — check `MemorySystem.health_check()` and the startup warnings before touching ranking weights.

## Conventions and traps

**`src/config.py` is the single source of truth for all tuning.** Nothing is hardcoded at call sites; every threshold, weight, and prefix is imported from it. The prose docs (`QUICK_REFERENCE.md`, `README.md`, `RESULTS_FEBRUARY_2026.md`) have drifted from it — e.g. they claim `MIN_SEMANTIC_SCORE = 0.3`, `MAX_MEMORIES_TO_RETRIEVE = 10`, `MEMORY_TOKEN_BUDGET = 500`, and a `stats['memories_extracted']` key, none of which match the code (`0.1`, `50`, `3000`, `stats['extracted_count']`). Read `config.py`, not the docs, and don't "fix" code to match a doc.

**Redis is not user-namespaced.** Keys are global (`mem:`, `type:`, `dedup:`, `recent_memories`) — only flat files (per-user directory) and Qdrant (`user_id` payload filter) isolate users. Consequences: two `MemorySystem` instances with different `user_id`s share the same Redis long-term store; `count_memories()` is global; `clear_memories()` wipes every user's memories **and** deletes the entire Qdrant collection. The evaluator calls `clear_memories()` per conversation, so never run an evaluation against a Redis/Qdrant instance holding data you care about.

**Adding a memory field is a four-place change.** Redis hashes can't hold `None` or `bool`, so the write path coerces (`None → ''`, `bool → str`) and `RedisStore.get_memory` re-casts field by field. A new field needs: `MEMORY_FIELDS` in config, a `setdefault` in `store_memory`, a cast in `get_memory`, and — if it should be searchable — the payload dict in `VectorStore.store_memory`. Skip the cast and downstream code gets a string where it expects a number.

**`MemorySystem.get_prompt_context` returns a tuple**, not the `str` its annotation and docstring promise — it forwards `_compose_prompt_context`'s `(context, active_memories)`. Callers in the demos unpack accordingly.

**Access counts are incremented twice per retrieval** — once in `retriever._retrieve_with_semantic_search` and again in `memory_system._compose_prompt_context`. It inflates the frequency signal (weighted 0.05, so effects are small). Fix the duplication rather than compensating in the weights.

**Repo hygiene:** `.gitignore` excludes `test_*.py` (whitelisting only `test_customer_conversation.py`), `memory/*/`, `output/`, and `*.log` — so `test_comprehensive_1000_turn.py` and per-user memory dirs are untracked by design. The README and QUICK_REFERENCE also reference scripts that don't exist in the repo (`demo.py` via `make demo`, `test_all_phases.py`, `test_1000_turn_latency.py`, `diagnostic_extraction_phases.py`); use the scripts listed under Commands above instead.
