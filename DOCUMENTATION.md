# Memora — Complete System Documentation

This document is a from-the-code description of how Memora actually works today. Where the existing README/QUICK_REFERENCE/ARCHITECTURE docs disagree with `src/config.py` or the implementation, this document follows the code and calls out the discrepancy — treat this as the accurate reference.

---

## Table of Contents

1. [What Memora Is](#1-what-memora-is)
2. [Core Concepts](#2-core-concepts)
3. [System Architecture](#3-system-architecture)
4. [Installation & Setup](#4-installation--setup)
5. [The Processing Pipeline](#5-the-processing-pipeline)
6. [Storage Layers in Depth](#6-storage-layers-in-depth)
7. [Extraction in Depth](#7-extraction-in-depth)
8. [Retrieval & Ranking in Depth](#8-retrieval--ranking-in-depth)
9. [Consolidation in Depth](#9-consolidation-in-depth)
10. [Configuration Reference](#10-configuration-reference)
11. [API Reference](#11-api-reference)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Multi-User Behavior & Isolation Gaps](#13-multi-user-behavior--isolation-gaps)
14. [Known Bugs & Sharp Edges](#14-known-bugs--sharp-edges)
15. [Module Reference Table](#15-module-reference-table)

---

## 1. What Memora Is

Memora is a Python library — not a service — that gives an LLM-based agent long-term memory across a conversation. It is not a chatbot and does not call an LLM to generate replies; it only manages *what to remember* and *what to inject into the next prompt*. The calling application is responsible for:

1. Calling `MemorySystem.process_turn(user_message)` before generating a response.
2. Taking the returned `memory_context` string and putting it into the LLM prompt (e.g., as a system-message prefix).
3. Generating the actual reply with its own LLM call.

Internally it maintains its own LLM client (Groq/OpenAI/Anthropic) purely for **extraction** — deciding what facts are worth remembering from a message — which is a separate concern from response generation.

The codebase describes itself in terms of five build phases, and that vocabulary appears throughout code comments, `config.py` section headers, and file docstrings. Phases are **not folders** — each phase adds capability to the same files:

| Phase | Capability added |
|---|---|
| **1** | Flat-file "core memory", Redis-backed long-term memory, two-stage extraction (heuristic + regex), type/recency retrieval |
| **2** | Qdrant vector store, `sentence-transformers` embeddings, semantic search, 3-signal ranking |
| **3** | LLM-based extraction (Stage 3), semantic deduplication, memory superseding/updates, confidence modifiers |
| **4** | Background consolidation (decay, merge, promote to core), 5-signal ranking, access-count tracking |
| **5** | `evaluation/` — a RAGAS-flavored evaluation harness with a synthetic conversation generator |

Package version: `src/__init__.py` → `__version__ = '0.2.0'` (the docs elsewhere claim "2.0.0" / "2.0.1" — these refer to documentation/changelog versioning, not the installed package version).

---

## 2. Core Concepts

### Memory record

Every extracted fact is a dictionary with a fixed schema (`MEMORY_FIELDS` in `config.py`):

```python
{
    "memory_id":          "mem_a1b2c3d4",     # or "merged_<hash>" for consolidation merges
    "type":                "preference",       # one of MEMORY_TYPES
    "key":                 "backend_language", # short machine key
    "value":               "Python",           # the actual content
    "confidence":          0.85,               # 0.0-1.0
    "turn_number":         42,                 # conversation turn it was created on
    "timestamp":           1735500000.0,       # unix time
    "source_text":         "I prefer Python for backend work.",
    "mention_count":       1,                  # incremented on re-mention / dedup hit
    "superseded_by":       None,               # memory_id of the update that replaced this one
    "supersedes":          None,               # memory_id this one replaced
    "is_update":           False,
    "last_accessed_turn":  42,
    "access_count":        0,                  # Phase 4: retrieval count
    "merged_from":         '',                 # Phase 4: JSON list of source ids, if merged
    "promoted_to_core":    False,               # Phase 4: True once written to a flat file
    "decay_applied":       0.0,                # Phase 4: cumulative confidence lost to decay
    "user_id":             "alice",            # added by MemorySystem.process_turn before storage
}
```

### Memory types

```
MEMORY_TYPES = ["preference", "constraint", "entity", "instruction", "commitment", "fact", "event"]
```

Each type has a **priority weight** (`TYPE_PRIORITIES`) used in ranking, and a subset is **promotable** to core memory (`PROMOTABLE_TYPES = ["entity", "preference", "constraint", "instruction"]`).

| Type | Priority | Meaning | Promotable |
|---|---|---|---|
| `constraint` | 1.0 | Hard rules the agent must never violate | ✅ |
| `instruction` | 0.95 | Behavioral/style directives | ✅ |
| `commitment` | 0.8 | Time-bound promises/deadlines | ✗ |
| `preference` | 0.7 | Soft likes/dislikes | ✅ |
| `entity` | 0.6 | People, places, organizations | ✅ |
| `fact` | 0.5 | General factual statements (includes all payment-domain extractions) | ✗ |
| `event` | 0.4 | Things that happened | ✗ |

### Two memory tiers

- **Core memory** — small, human-editable, **always** injected into every prompt regardless of relevance. Lives as Markdown files on disk, one directory per user.
- **Long-term memory** — potentially thousands of records, selectively retrieved per-turn based on relevance, stored in Redis (source of truth) and mirrored into Qdrant (for semantic search only).

Consolidation is the bridge between the two: memories that prove durably important (high confidence, frequently mentioned, frequently accessed, old enough) get **promoted** out of long-term memory and appended into a core-memory file, so they're always in context from then on.

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                            MemorySystem                               │
│                     (src/memory_system.py — orchestrator)             │
└───────────────────────────────────────────────────────────────────────┘
        │                    │                    │                │
        ▼                    ▼                    ▼                ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────┐  ┌──────────────────┐
│ FlatFileStore  │   │ MemoryExtractor │   │ MemoryRetriever│  │ ConsolidationWorker │
│ (core memory)  │   │ (3-stage        │   │ (hybrid +      │  │ (decay/merge/     │
│                │   │  extraction)     │   │  5-signal rank)│  │  promote)          │
└───────┬────────┘   └────────┬────────┘   └───────┬────────┘  └─────────┬──────────┘
        │                     │                     │                     │
        ▼                     ▼                     ▼                     │
┌───────────────┐   ┌────────────────┐   ┌───────────────┐                │
│ memory/<user>/ │   │  LLMExtractor   │   │  RedisStore    │◄───────────────┘
│  *.md files    │   │ (Stage 3, lazy) │   │ (Phase 1 truth)│
└───────────────┘   └────────┬────────┘   └───────┬────────┘
                              │                     │
                              ▼                     ▼
                     ┌────────────────┐   ┌───────────────────┐
                     │ Groq/OpenAI/    │   │  VectorStore       │
                     │ Anthropic API   │   │  (Qdrant, Phase 2)  │
                     └────────────────┘   └──────────┬─────────┘
                                                       ▼
                                              ┌───────────────────┐
                                              │ EmbeddingService    │
                                              │ (sentence-          │
                                              │  transformers)      │
                                              └───────────────────┘
```

**Every optional layer degrades gracefully.** `src/__init__.py`, `MemorySystem.__init__`, and `MemoryExtractor.__init__` each wrap their imports of Phase 2/3/4 components in `try/except ImportError`, logging a warning and falling back rather than crashing:

- No Qdrant / no `sentence-transformers` → `_semantic_enabled = False`, retrieval falls back to `MemoryRetriever._retrieve_phase1` (type-priority + recency only, no embeddings needed).
- No consolidation worker importable → `_consolidation_enabled = False`, `process_turn` simply never calls it.
- No Stage-3 LLM extractor (missing API key / package) → `extractor.stage3_enabled` still True but `llm_extractor` is `None`, so Stage 3 is silently skipped and everything falls back to Stage 2 regex output.

This means a "why isn't semantic search finding anything" bug report is very often just Qdrant being unreachable — always check `MemorySystem.health_check()` first.

---

## 4. Installation & Setup

### Prerequisites

- Python 3.8+
- Docker + Docker Compose (for Redis and Qdrant)
- An LLM API key for Stage-3 extraction: Groq (recommended — supports free-tier multi-key rotation), OpenAI, or Anthropic

### Steps

```bash
pip install -r requirements.txt
docker-compose up -d          # starts redis:7-alpine (6379) and qdrant/qdrant:v1.7.4 (6333/6334)
docker-compose ps             # verify both containers are healthy
```

Copy `.env.example` to `.env` and set at minimum one LLM key:

```bash
# .env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=memory_vectors
LLM_PROVIDER=groq
LLM_EXTRACTION_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk-...
# Optional: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3 for rate-limit rotation
LOG_LEVEL=INFO
```

`src/config.py` loads `.env` from the project root via `python-dotenv` at import time (`env_path = Path(__file__).parent.parent / ".env"`), so `.env` must sit next to `docker-compose.yml`, not inside `src/`.

### Verifying the install

```python
from src import MemorySystem
ms = MemorySystem(user_id="smoke_test")
print(ms.health_check())
# {'redis': True, 'flat_files': True, 'vector_store': True}
```

### Redis persistence

`redis.conf` is mounted into the Redis container and configures AOF (append-only file) persistence, so memories survive container restarts as long as the `redis_data` Docker volume isn't removed (`docker-compose down -v` *would* wipe it).

---

## 5. The Processing Pipeline

`MemorySystem.process_turn(user_message, priority_types=None)` is the single entry point. Each call executes, in order:

```
user_message
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. EXTRACT   (MemoryExtractor.extract)                        │
│    Stage 1 heuristic filter → Stage 2 regex → Stage 3 LLM      │
│    (Stage 3 only if Stage 2 was empty/low-confidence)          │
└─────────────────────────────────┬───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DEDUP + STORE  (per extracted memory)                       │
│    - semantic-dup check (Qdrant, same type+user, ≥0.92 sim)    │
│      → duplicate & not an update  → boost confidence, skip     │
│      → duplicate & is_update       → store new, supersede old  │
│      → no duplicate                → store as new              │
│    - store_memory() writes Redis hash + 3 indices               │
│    - store_memory() also writes into Qdrant (if enabled)        │
└─────────────────────────────────┬───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RETRIEVE + INJECT  (MemoryRetriever.retrieve + format)       │
│    - core memory (flat files) is unconditionally read           │
│    - long-term memory retrieved via hybrid ranking               │
│    - both concatenated into memory_context string                │
│    - access_count incremented for each retrieved memory (2x, see │
│      §14 known bugs)                                              │
└─────────────────────────────────┬───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CONSOLIDATE (conditional)                                   │
│    Every CONSOLIDATION_INTERVAL_TURNS turns (default 50), and   │
│    only if ≥ CONSOLIDATION_MIN_MEMORIES exist, runs inline       │
│    (synchronously — this is NOT a background thread despite      │
│    the class name "ConsolidationWorker")                          │
└─────────────────────────────────┬───────────────────────────┘
                                    ▼
                    return (memory_context, stats)
```

`stats` is a dict accumulated across all four steps — see [§11 API Reference](#11-api-reference) for its exact shape.

---

## 6. Storage Layers in Depth

### 6.1 Flat-file core memory (`src/flat_file_store.py`)

- One directory per user: `memory/<user_id>/`.
- Four fixed files (`CORE_MEMORY_FILES` in config): `CORE.md`, `PREFERENCES.md`, `INSTRUCTIONS.md`, `CONSTRAINTS.md`.
- On first construction of `FlatFileStore(user_id)`, any missing file is created from a template containing a `## <Section>` header (`Identity`, `General Preferences`, `Communication Style`, `Hard Constraints` respectively) plus a `**Last Updated:**` line. This is important: **the promotion feature depends on these exact headers existing**, because `append_to_section` locates the insertion point by scanning for a line starting with `## ` that contains the section name. If you hand-edit these files and remove/rename a header, promotion for that type will log `"Section {section} not found"` and silently no-op.
- `read_core_memory()` concatenates all four files (skipping empty ones) with `=== filename.md ===` separators — this is what's "always injected."
- `update_core_field(file, section, field, value)` does a targeted find-and-replace of a line matching `- **Field:**` inside a section — meant for high-confidence, rare identity corrections, not bulk writes.
- `append_to_section(file, section, content)` inserts a new bullet at the end of a section — this is what `ConsolidationWorker._promote_to_core` calls.

### 6.2 Redis long-term memory (`src/redis_store.py`)

Redis is the **source of truth** for long-term memory; Qdrant is a derived index. Key layout:

| Key pattern | Type | Purpose |
|---|---|---|
| `mem:<memory_id>` | Hash | Full memory record (all `MEMORY_FIELDS`) |
| `dedup:<type>:<key>` | String | Maps `(type, key)` → `memory_id`, for exact-key dedup at write time |
| `type:<type>` | Set | All memory_ids of a given type, for `get_memories_by_type` |
| `recent_memories` | Sorted Set (score = timestamp) | Global recency ordering; also the basis for `count_memories()` |

Notable behaviors:

- **Redis hashes can't store `None` or Python `bool`.** `store_memory` coerces `None → ''` and `bool → str(bool)` before writing; `get_memory` reverses this on read (casting `'True'/'true'/'1'` back to `True`, and empty string back to `None` for `superseded_by`/`supersedes`). Any new boolean/optional field added later must be added to both coercion paths or it'll come back as a raw string.
- **Exact-key dedup happens here, before semantic dedup.** If a `(type, key)` pair already has a `dedup:` entry, `store_memory` returns `False` immediately — it doesn't even reach the semantic-dedup check in `memory_system.py`. This is why extremely literal repeats (e.g. two Stage-2 matches producing the identical key) never generate near-duplicate confidence boosts twice; the second one just bumps `mention_count`/`confidence` via `_update_recency` + `boost_confidence`.
- `supersede_memory(old_id, new_id)` sets `superseded_by` on the old record and `supersedes` on the new one — retrieval later filters out anything with a non-empty `superseded_by`.
- `clear_all_memories()` iterates every id in `recent_memories` and calls `delete_memory` on each (properly removing all four index entries) — it does **not** do a Redis `FLUSHDB`, so it only touches memory-related keys.

### 6.3 Vector store (`src/vector_store.py` + `src/embedding_service.py`)

- Qdrant collection name from `QDRANT_COLLECTION_NAME` (default `memory_vectors`), 384-dim vectors, cosine distance, with a payload index on `memory_type` for filtered search.
- `_hash_id(memory_id)` converts the string memory id into an unsigned 64-bit int via MD5 truncation, since Qdrant point ids must be int or UUID — this hash is stable across runs (same string always maps to same int) but is a plain truncated hash, not collision-proof (astronomically unlikely to matter at this scale).
- Embeddings are computed via **sentence-transformers `all-MiniLM-L6-v2`**, lazily loaded as a module-level singleton on first use (`embedding_service._get_model()`), so the first embed call in a process takes several seconds (model download/load) — subsequent calls are fast.
- `EmbeddingService.embed_memory(memory)` builds the embedded text as `"{key} | {value} | type: {type}"` — so semantic search is over key+value+type jointly, not the raw source sentence.
- The Qdrant payload stores a **subset** of memory fields (`memory_id`, `user_id`, `memory_type`, `key`, `value`, `confidence`, `turn_number`, `timestamp`) — no `mention_count`, `access_count`, `superseded_by`, etc. Retrieval therefore always re-hydrates the full record from Redis by id (`retriever.py`'s semantic branch calls `redis_store.get_memory(memory_id)` and only falls back to the thin Qdrant payload if Redis has lost the record).
- `search_similar` / `search_by_vector` accept an optional `user_id` filter (`models.FieldCondition` on the `user_id` payload field) — this is the *only* mechanism providing multi-user isolation in semantic search (see §13).
- `clear()` does `delete_collection` + `_ensure_collection()` — i.e. it drops and recreates the **entire** collection, not just points for one user.

### 6.4 Storage layer comparison

| | Flat files | Redis | Qdrant |
|---|---|---|---|
| Contents | Core identity Markdown | Full memory records + indices | Embeddings + thin payload |
| Authoritative? | Yes (its own tier) | Yes, for long-term memory | No — a derived cache |
| User-isolated? | Yes (own directory) | **No** — global keyspace | Yes (`user_id` payload filter) |
| Survives Qdrant outage? | N/A | Yes, fully functional | — |
| Survives Redis outage? | N/A | — | System can't function — Redis holds all field data |

---

## 7. Extraction in Depth

`src/extractor.py` — a 3-stage escalating cascade. Because each stage is cheap-to-expensive, most turns never reach the LLM (~87% pattern-only per `RESULTS_FEBRUARY_2026.md`'s own measurement, though see §14 about doc/code drift on other numbers).

### Stage 1 — Sensory filter (`should_extract`)

Purely heuristic, no regex matching yet. Rejects immediately if the message is under 5 characters or is an exact match to a greeting/ack list (`"hi", "hello", "hey", "thanks", "ok", "okay", "cool", "nice"`). Otherwise computes a weighted score from `HEURISTIC_WEIGHTS`:

```
score = 0.3 * min(len(msg)/100, 1)                       # length
      + 0.4 * min(keyword_hits/3, 1)                     # EXTRACTION_KEYWORDS hit count
      + 0.15 * (1 if "?" in msg else 0)                   # question marker
      + 0.15 * min(specificity_hits/3, 1)                 # digits, capitalized words, am/pm, "@"
```

Passes if `score >= SENSORY_FILTER_THRESHOLD` (0.3).

### Stage 2 — Regex classifier (`classify_and_extract`)

A fixed set of pattern tables, each `(regex, key_template, base_confidence)`, one table per memory type: preference, constraint, entity, commitment, instruction, plus a dedicated **payment/financial** block (13 patterns covering account numbers, payment amounts, due dates, payment status, arrangements, and speaker/customer name in a support-call context — all stored as `type="fact"`).

Important asymmetry: most tables match against `message.lower()`, but the payment patterns match against the **raw** `message` (unlowercased) — they rely on capitalization to recognize proper-noun names and month names (e.g. `r"due (?:on |date )?([A-Z][a-z]+ \d+(?:st|nd|rd|th)?)"`). Lowercasing the input before calling this stage would silently break every payment-domain pattern.

Every match runs through `_apply_confidence_modifiers`, which nudges the base confidence up/down based on certainty words present anywhere in the source text (`CONFIDENCE_MODIFIERS`: `"always"/"never"/"definitely"/"absolutely"/"must"` → `+0.1`; `"maybe"/"perhaps"/"possibly"/"might"` → `−0.2`; `"sometimes"/"occasionally"/"could"` → `−0.15`), clamped to `[0, 1]`.

### Escalation decision

Stage 3 is invoked only when:
- Stage 2 produced **nothing** *and* the Stage-1 heuristic score was `> 0.5` (message still "looks important"), **or**
- Stage 2 produced results but the **best** confidence among them is `< STAGE_3_CONFIDENCE_THRESHOLD` (0.7).

### Stage 3 — LLM extraction (`src/llm_extractor.py`)

- One shared prompt template (`LLMExtractor.EXTRACTION_PROMPT`) with few-shot examples covering entities, preferences, updates (`is_update: true`), constraints, and low-confidence hedged statements ("maybe I'll try Python").
- Includes the last 3 turns of conversation as context (`extractor.context_buffer`, capped at 3 via `pop(0)`), plus an optional hint string derived from Stage 2's detected type if Stage 2 fired at low confidence.
- Provider selection is config-driven (`LLM_PROVIDER = "groq" | "openai" | "anthropic"`), each with its own lazily-constructed client.
- **Groq multi-key rotation**: `GROQ_API_KEYS` collects up to 4 keys (`GROQ_API_KEY`, `_1`, `_2`, `_3`) from the environment. Each key gets its own `Groq(..., max_retries=0)` client (SDK-level retries are disabled on purpose so the rotation logic — not the SDK — decides what counts as retryable). `_call_groq` loops over all clients on a 429/rate-limit error, rotating via a module-level `_current_groq_key_index`; any non-429 exception is re-raised immediately without rotation.
- Response parsing (`_parse_and_validate`) strips Markdown code fences, `json.loads`, validates each item has `type/key/value/confidence`, clamps confidence to `[0,1]`, and assigns a synthetic `memory_id` (`mem_{turn}_{index}`, **not** a UUID like Stage 2/config templates — worth noting if code elsewhere assumes the `mem_<8-hex>` shape). On a JSON parse failure, it does **one** retry with an error-feedback prompt before giving up and returning `[]`.
- `detect_update_intent(message)` — a standalone helper matching `UPDATE_PATTERNS` (regexes like `"actually, ..."`, `"i changed my mind"`, `"correction:"`) — exists on the class but is not currently called anywhere in the extraction pipeline; update detection in practice comes from the LLM setting `is_update: true` in its own output, or from Stage 2 not covering updates at all (Stage 2 has no update-detection logic).

### Merging Stage 2 + Stage 3 results

`MemoryExtractor._merge_stage_results` unions both lists keyed by `(type, key)`; where both stages produced the same key, the higher-confidence one wins. Final output is filtered once more by `MIN_CONFIDENCE_TO_STORE` (0.6) before returning from `extract()`.

---

## 8. Retrieval & Ranking in Depth

`src/retriever.py` — `MemoryRetriever.retrieve()` dispatches to one of two implementations depending on whether semantic search is available.

### 8.1 Hybrid retrieval (semantic search enabled) — the default path

This is a **dual-branch** design specifically built to fix a failure mode where old-but-relevant memories with weak lexical/semantic overlap to the current query would never surface:

1. **Semantic branch** — `vector_store.search_similar(query, limit=SEMANTIC_SEARCH_LIMIT=100, min_score=MIN_SEMANTIC_SCORE=0.1, user_id=...)`. Each hit is re-hydrated from Redis (falls back to the Qdrant payload only if Redis lookup fails).
2. **Recency branch** (only if `HYBRID_RETRIEVAL_ENABLED`) — `redis_store.get_recent_memories(limit=RECENCY_RETRIEVAL_LIMIT=50)`, added for any memory_id not already present, with an assigned neutral `semantic_score = RECENCY_FALLBACK_SEMANTIC_SCORE (0.15)` so recency-only candidates don't dominate ranking purely by getting a fake high semantic score.
3. **Always-on types** — `constraint` and `instruction` are unconditionally fetched (`get_memories_by_type`, limit 20 each) and merged in, so critical rules can never be filtered out by a poor semantic match.
4. **Priority types** (optional `priority_types` argument) — additional types the caller wants included regardless of relevance, added with a low default score (0.3) if not already present.
5. **Ranking** — every candidate gets a `final_score` from the 5-signal weighted sum (see below), the merged pool is sorted descending, superseded memories are filtered out, and the result is truncated to `MAX_MEMORIES_TO_RETRIEVE` (50) then to whatever fits `MEMORY_TOKEN_BUDGET` (3000) at a rough estimate of 50 tokens/memory.
6. **Access tracking** — every memory in the final top-K has `redis_store.increment_access_count(memory_id, turn_number)` called on it *inside this method* — see §14 for the duplicate-increment bug.

#### 5-signal ranking formula

```
final_score = 0.30 * semantic_score      # cosine similarity to query (0 for recency-only additions' fallback... see note)
            + 0.40 * type_score          # TYPE_PRIORITIES[type]
            + 0.10 * recency_score       # exp(-RECENCY_DECAY_RATE * turns_ago), RECENCY_DECAY_RATE=0.001
            + 0.05 * frequency_score     # access-count based, see below
            + 0.15 * confidence_score    # the memory's own stored `confidence` field
```

`frequency_score` combines normalized access count and recency-of-last-access:

```
normalized_access   = min(1, access_count / FREQUENCY_MAX_ACCESSES=20)
access_recency_decay = exp(-FREQUENCY_DECAY_RATE=0.05 * (turn_number - last_accessed_turn))
frequency_score = (1 - ACCESS_RECENCY_WEIGHT=0.6) * normalized_access
                +      ACCESS_RECENCY_WEIGHT       * access_recency_decay * normalized_access
```

Weight rationale documented in `config.py` comments: semantic weight was deliberately reduced (from an earlier higher value) because generic queries were letting semantic similarity dominate and starve out constraint/instruction diversity; frequency was reduced to near-zero because it's circular — memories that get retrieved gain access_count, which makes them more likely to be retrieved again, compounding regardless of actual relevance.

### 8.2 Phase-1 fallback (no semantic search)

`_retrieve_phase1` — used automatically if Qdrant/embeddings aren't available. Much simpler: always-on types get `retrieval_score = 1.0`; the 30 most recent memories get `0.9^i * 0.5` (exponential positional decay, i.e. the very newest gets 0.45, decaying fast); explicit `priority_types` get a flat `0.7` if not already present. Deduplicated by keeping the highest score per `memory_id`, sorted, superseded-filtered, truncated the same way as the hybrid path. This path does **not** call `increment_access_count` at all (that only happens in `_retrieve_with_semantic_search` and, redundantly, in `memory_system.py`).

### 8.3 Prompt formatting

`format_memories_for_prompt` groups retrieved memories by type into fixed sections (`=== CONSTRAINT ===`, `=== INSTRUCTION ===`, etc., in the type-priority order above) and renders each as:

```
- {key}: {value} [turn {turn_number}, {confidence*100:.0f}% confident]
```

The final `memory_context` returned to the caller is `"### CORE MEMORY (Always Active)\n\n{core}\n\n### LONG-TERM MEMORY (Retrieved)\n\n{formatted}"` (either section omitted if empty).

---

## 9. Consolidation in Depth

`src/consolidation_worker.py` — despite the name, this runs **synchronously inline** inside `process_turn`, not on a background thread or separate process. `MemorySystem` checks `consolidation_worker.needs_consolidation(turn_number, CONSOLIDATION_INTERVAL_TURNS=50)` every turn; that returns true either on the very first run once `count_memories() >= CONSOLIDATION_MIN_MEMORIES (10)`, or every 50 turns thereafter. `run_consolidation` itself also independently gates on the minimum-memory count unless `force=True`.

Three independently-flagged sub-operations run in sequence, each contributing to a combined stats dict:

### 9.1 Decay (`MEMORY_DECAY_ENABLED` — currently `False` by default)

For every non-promoted memory older than `DECAY_TURNS_THRESHOLD` (500 turns):

```
decay = (turns_old / 100) * DECAY_RATE_PER_100_TURNS(0.005)
if turns_since_last_access > DECAY_INACTIVE_TURNS(300):
    decay += ((turns_since_access - 300) / 100) * DECAY_RATE_PER_100_TURNS * 0.5
new_confidence = max(0.1, confidence - decay)
```

If `new_confidence < MIN_DECAY_CONFIDENCE(0.05)` and `DELETE_VERY_LOW_CONFIDENCE` is set, the memory is deleted outright from both Redis and Qdrant. Otherwise `confidence` and cumulative `decay_applied` are updated in place. The config comments note this rate was deliberately slowed ~20x from an earlier, too-aggressive setting, and decay is currently disabled entirely so memories can survive long enough to satisfy promotion's age/access thresholds during testing — re-enable deliberately, understanding it will compete with promotion for the same memories.

### 9.2 Merge (`MEMORY_MERGE_ENABLED`, requires a vector store)

For each unmerged, unsuperseded, unpromoted memory, searches Qdrant for others with similarity `≥ MERGE_SIMILARITY_THRESHOLD (0.85)` — **note this Qdrant search does not pass a `user_id` filter** (see §13 for the isolation implication) — and, subject to `MERGE_SAME_TYPE_ONLY`, merges the first qualifying match:

- Higher-confidence memory becomes "primary"; its `key` is kept.
- Values are combined: identical → keep one; one is a substring of the other → keep the longer; otherwise concatenate as `"{val1}; {val2}"`, truncated to `MAX_MERGED_VALUE_LENGTH` (500 chars).
- New id is `merged_{md5(id1_id2_turn)[:8]}`, confidence is `min(MAX_CONFIDENCE, avg(conf1,conf2) + 0.05)`, mention counts and access counts are summed, `merged_from` records both source ids as JSON.
- Both originals are marked `superseded_by` the new merged record (so retrieval's superseded-filter removes them going forward), and the merged record is stored fresh in both Redis and Qdrant.
- Each memory only participates in at most one merge per consolidation pass (`break` after first match; `merged_ids` tracked to avoid reprocessing).

Distinct from **semantic deduplication** (Phase 3, runs at write time, threshold 0.92, discards the new duplicate or supersedes on explicit update) — merging runs at consolidation time on a lower 0.85–0.92 similarity band, combining two already-stored memories into one rather than rejecting a new write.

### 9.3 Promotion (`PROMOTION_ENABLED`)

A memory is promoted to core memory once **all** of the following hold:
- `confidence >= PROMOTION_CONFIDENCE_THRESHOLD (0.90)`
- `mention_count >= PROMOTION_MENTION_THRESHOLD (3)`
- `access_count >= PROMOTION_ACCESS_THRESHOLD (5)`
- `turn_number_age >= PROMOTION_AGE_THRESHOLD (50)`
- `type in PROMOTABLE_TYPES` (`entity`, `preference`, `constraint`, `instruction`)
- not already promoted, not superseded

`_get_promotion_target(type)` maps type → `(file, section)`:

```
entity      → CORE.md        / "Identity"
preference  → PREFERENCES.md / "General Preferences"
constraint  → CONSTRAINTS.md / "Hard Constraints"
instruction → INSTRUCTIONS.md / "Communication Style"
```

The content line written is `**{key}:** {value}`, appended via `FlatFileStore.append_to_section` (which depends on the `##` header text matching exactly — see §6.1). On success the Redis field `promoted_to_core` is set to `"True"`; the memory itself is **not** deleted from Redis/Qdrant — it remains retrievable as long-term memory in addition to now being always-injected as core memory (i.e., promotion duplicates the fact into both tiers rather than moving it).

---

## 10. Configuration Reference

All tunables live in `src/config.py`, loaded once at import time. Nothing in the codebase hardcodes a threshold at the call site — always change behavior here, not inline.

### Storage endpoints
| Setting | Default | Source |
|---|---|---|
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `localhost` / `6379` / `0` | env |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION_NAME` | `localhost` / `6333` / `memory_vectors` | env |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `all-MiniLM-L6-v2` / `384` | fixed |

### Core memory
| Setting | Default |
|---|---|
| `CORE_MEMORY_FILES` | `["CORE.md", "PREFERENCES.md", "INSTRUCTIONS.md", "CONSTRAINTS.md"]` |
| `CORE_MEMORY_TOKEN_BUDGET` | `500` (documented target; not currently enforced in code) |

### Extraction (Stage 1/2)
| Setting | Default |
|---|---|
| `SENSORY_FILTER_THRESHOLD` | `0.3` |
| `EXTRACTION_CLASSIFIER_THRESHOLD` | `0.6` (defined but not directly referenced by extractor logic — Stage 2 confidence comes straight from each pattern table) |
| `HEURISTIC_WEIGHTS` | length 0.3 / keywords 0.4 / question 0.15 / specificity 0.15 |
| `MEMORY_TYPES` | 7 types listed in §2 |

### Retrieval
| Setting | Default |
|---|---|
| `MAX_MEMORIES_TO_RETRIEVE` | `50` |
| `MEMORY_TOKEN_BUDGET` | `3000` |
| `SEMANTIC_SEARCH_ENABLED` | `True` |
| `SEMANTIC_SEARCH_LIMIT` | `100` |
| `MIN_SEMANTIC_SCORE` | `0.1` |
| `HYBRID_RETRIEVAL_ENABLED` | `True` |
| `RECENCY_RETRIEVAL_LIMIT` | `50` |
| `RECENCY_FALLBACK_SEMANTIC_SCORE` | `0.15` |
| `RANKING_WEIGHTS` (3-signal, legacy path) | semantic 0.5 / type 0.25 / recency 0.25 |
| `TYPE_PRIORITIES` | see §2 table |
| `RECENCY_DECAY_RATE` / `RECENCY_MAX_TURNS` | `0.001` / `5000` |
| `RANKING_WEIGHTS_5_SIGNAL` | semantic 0.30 / type 0.40 / recency 0.10 / frequency 0.05 / confidence 0.15 |
| `FREQUENCY_DECAY_RATE` / `FREQUENCY_MAX_ACCESSES` / `ACCESS_RECENCY_WEIGHT` | `0.05` / `20` / `0.6` |

### Stage 3 (LLM extraction)
| Setting | Default |
|---|---|
| `STAGE_3_ENABLED` | `True` |
| `LLM_PROVIDER` | `groq` |
| `LLM_EXTRACTION_MODEL` | env `LLM_EXTRACTION_MODEL`, default `llama-3.3-70b-versatile` |
| `GROQ_API_KEYS` | up to 4 keys from `GROQ_API_KEY(_1/_2/_3)` env vars |
| `STAGE_3_CONFIDENCE_THRESHOLD` | `0.7` |
| `STAGE_3_MAX_TOKENS` / `STAGE_3_TEMPERATURE` | `500` / `0.1` |

### Confidence & deduplication
| Setting | Default |
|---|---|
| `MIN_CONFIDENCE_TO_STORE` | `0.6` |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.9` (defined; not directly referenced — promotion uses `PROMOTION_CONFIDENCE_THRESHOLD` instead) |
| `CONFIDENCE_BOOST_PER_MENTION` / `MAX_CONFIDENCE` | `0.1` / `0.95` |
| `SEMANTIC_DEDUP_ENABLED` / `_THRESHOLD` / `_CHECK_LIMIT` | `True` / `0.92` / `5` |
| `CONFIDENCE_MODIFIERS` | certainty words → ±confidence, see §7 |
| `UPDATE_PATTERNS` | regexes for `detect_update_intent` (currently unused in the live pipeline) |

### Consolidation (Phase 4)
| Setting | Default |
|---|---|
| `CONSOLIDATION_ENABLED` / `_INTERVAL_TURNS` / `_MIN_MEMORIES` | `True` / `50` / `10` |
| `MEMORY_DECAY_ENABLED` | **`False`** (intentionally disabled — see §9.1) |
| `DECAY_TURNS_THRESHOLD` / `DECAY_INACTIVE_TURNS` / `DECAY_RATE_PER_100_TURNS` / `MIN_DECAY_CONFIDENCE` / `DELETE_VERY_LOW_CONFIDENCE` | `500` / `300` / `0.005` / `0.05` / `True` |
| `MEMORY_MERGE_ENABLED` / `MERGE_SIMILARITY_THRESHOLD` / `MERGE_SAME_TYPE_ONLY` / `MAX_MERGED_VALUE_LENGTH` | `True` / `0.85` / `True` / `500` |
| `PROMOTION_ENABLED` / `_CONFIDENCE_THRESHOLD` / `_MENTION_THRESHOLD` / `_ACCESS_THRESHOLD` / `_AGE_THRESHOLD` | `True` / `0.90` / `3` / `5` / `50` |
| `PROMOTABLE_TYPES` | `["entity", "preference", "constraint", "instruction"]` |

### Redis key layout
```
REDIS_MEMORY_PREFIX      = "mem:"
REDIS_DEDUP_PREFIX       = "dedup:"
REDIS_TYPE_INDEX_PREFIX  = "type:"
REDIS_RECENCY_INDEX      = "recent_memories"
```

**A note on stale documentation:** `QUICK_REFERENCE.md` currently states `MIN_SEMANTIC_SCORE = 0.3`, `MAX_MEMORIES_TO_RETRIEVE = 10`, `MEMORY_TOKEN_BUDGET = 500`. These do not match `config.py` (`0.1`, `50`, `3000` respectively) — they reflect an earlier tuning pass. This document and `config.py` are the sources of truth.

---

## 11. API Reference

### `MemorySystem(user_id, enable_semantic_search=None, json_log_path=None)`

Constructs all storage/processing layers for one user. `enable_semantic_search=None` defers to `config.SEMANTIC_SEARCH_ENABLED`; pass `False` to force Phase-1-only retrieval even if Qdrant is reachable. `json_log_path`, if given, causes every `process_turn` call to append to and rewrite a JSON file with the full turn-by-turn stats history (see `example_json_logging.py`).

#### `process_turn(user_message: str, priority_types: Optional[List[str]] = None) -> Tuple[str, Dict]`

The main entry point. Returns `(memory_context, stats)`.

`stats` keys:
```python
{
    "turn_number": int,
    "extracted_count": int, "stored_count": int, "retrieved_count": int,
    "total_memories": int,                       # global Redis count, not per-user
    "extraction_time_ms": float, "storage_time_ms": float, "retrieval_time_ms": float,
    "response_generated": True,                   # static placeholder, not a real signal
    "vector_stored_count": int, "dedup_count": int, "superseded_count": int,
    "semantic_enabled": bool,
    "active_memories": [                          # one entry per retrieved memory
        {
            "memory_id": str, "content": "key: value", "type": str,
            "origin_turn": int, "last_used_turn": int,   # = current turn_number
            "confidence": float, "mention_count": int, "access_count": int,
        }, ...
    ],
    "consolidation": Optional[Dict],              # present only on turns that trigger it
}
```

#### `get_prompt_context(user_message, priority_types=None) -> Tuple[str, List[Dict]]`

**Despite its `-> str` type hint and docstring**, this returns the same 2-tuple as the internal `_compose_prompt_context` — `(memory_context, active_memories_list)`, not a bare string. Use it to preview what would be retrieved for a message **without** running extraction/storage (useful for testing retrieval tuning in isolation).

#### `update_core_memory(file: str, section: str, field: str, value: str) -> None`

Thin wrapper over `FlatFileStore.update_core_field`. Intended for rare, high-confidence identity corrections, not routine writes.

#### `get_statistics() -> Dict`

```python
{
    "total_turns": int, "total_memories": int,           # global Redis count
    "memories_by_type": {type: count, ...},               # only non-zero types included
    "extraction_count": int,                              # cumulative across the extractor's lifetime
    "semantic_search_enabled": bool,
    "vector_count": int,                                  # only present if semantic search enabled; -1 on error
}
```

#### `clear_memories() -> None`

Deletes **every** memory in Redis (all users) and, if semantic search is enabled, drops and recreates the entire Qdrant collection. See §13 — this is global, not per-user.

#### `health_check() -> Dict[str, Optional[bool]]`

`{"redis": bool, "flat_files": bool, "vector_store": bool | None}` — `None` means semantic search isn't enabled for this instance, not that the check failed.

#### Consolidation controls

- `run_consolidation(force: bool = False) -> Optional[Dict]` — manually trigger a consolidation cycle outside the normal interval; returns `None` if the worker isn't enabled.
- `get_consolidation_stats() -> Optional[Dict]` — last-run counters (`decayed`, `deleted`, `merged`, `promoted`, `last_run` ISO timestamp).
- `is_consolidation_enabled() -> bool`

#### JSON logging

- `get_turn_stats_log() -> List[Dict]` — in-memory copy of every stats dict produced so far this session.
- `export_turn_stats(output_path: str) -> None` — writes `{"user_id", "total_turns", "conversation_turns": [...]}` to an arbitrary path, independent of `json_log_path` (useful for a one-off snapshot even if continuous logging wasn't enabled at construction).

### Minimal usage example

```python
from src import MemorySystem

memory = MemorySystem(user_id="alice")

for user_message in conversation:
    memory_context, stats = memory.process_turn(user_message)

    prompt = f"{memory_context}\n\nUser: {user_message}\nAssistant:"
    response = your_llm_call(prompt)   # Memora does not do this part

    for mem in stats["active_memories"]:
        print(mem["content"], mem["confidence"])
```

---

## 12. Evaluation Framework

`evaluation/` (Phase 5) is a self-contained harness, importable independently of the demo scripts.

- **`conversation_generator.py`** (`ConversationGenerator(seed=42)`) — produces synthetic multi-turn conversations with attached ground-truth memory annotations, used to measure extraction/retrieval quality against a known-correct answer set. `generate_batch(num_conversations, turns_per_conversation, output_file)` and `generate_distance_sweep_conversation(target_distances=[...])` (the latter purpose-built to test recall at specific turn distances, e.g. "was a fact from turn 1 still recallable at turn 1000").
- **`metrics.py`** — four metric classes:
  - `ExtractionMetrics.evaluate(extracted, ground_truth)` — precision/recall/F1 via a loose matcher (`_memories_match`: same `type` plus substring containment on either `value` or `key`).
  - `RetrievalMetrics` — RAGAS-style `context_precision`/`context_recall` plus MRR-style ranking position.
  - `DistanceSweepMetrics` — recall as a function of how many turns ago the ground-truth fact was introduced.
  - `ConsolidationMetrics` — decay/merge/promotion effectiveness.
- **`evaluator.py`** (`MemorySystemEvaluator`) — orchestrates the above. `evaluate_extraction`/`evaluate_retrieval` each call `memory_system.clear_memories()` **before every single conversation** to reset state — this is why evaluation must never be pointed at a Redis/Qdrant instance holding data you care about (§13).
- **`run_evaluation.py`** — the CLI entry point: generates fixtures if `evaluation/fixtures/test_conversations.json` is absent, health-checks the system, runs the full suite, and prints a formatted summary (extraction P/R/F1, context precision/recall/MRR, per-distance recall, consolidation counts) plus writes `evaluation/results/evaluation_results.json`.

To run just one slice (e.g. while tuning ranking weights) without the full 200-conversation batch:

```python
from evaluation.evaluator import MemorySystemEvaluator
import json

conversations = json.load(open("evaluation/fixtures/test_conversations.json"))["conversations"][:10]
MemorySystemEvaluator().evaluate_retrieval(conversations, verbose=True)
```

There is no `pytest`-collected test suite in this repo — `requirements_evaluation.txt` lists `pytest`/`pytest-cov` as dependencies but no test functions exist for them to collect; `make test` is a documented stub ("Tests not yet implemented (Phase 5)"). Verification is done by running the demo/evaluation scripts and inspecting their printed output or JSON artifacts.

---

## 13. Multi-User Behavior & Isolation Gaps

Memora supports multiple users via `user_id`, but isolation is **inconsistent across layers** — worth understanding before relying on it in a multi-tenant deployment:

| Layer | Isolation mechanism | Isolated? |
|---|---|---|
| Flat files | Separate directory per `user_id` | ✅ Fully isolated |
| Redis | Global keyspace (`mem:`, `type:`, `dedup:`, `recent_memories` — none namespaced by user) | ❌ **Shared across all users** |
| Qdrant reads (retrieval, semantic-dedup check) | `user_id` payload filter passed explicitly | ✅ Isolated, when the caller remembers to pass `user_id` |
| Qdrant writes (merge-candidate search in consolidation) | No `user_id` filter passed | ❌ **Can match across users** |

Practical consequences:

- Two `MemorySystem("alice")` and `MemorySystem("bob")` instances **share the same Redis long-term memory pool**. `redis_store.count_memories()`, `get_recent_memories()`, and the always-on `constraint`/`instruction` type fetches in retrieval are **not** filtered by user — Alice's constraints can be injected into Bob's prompt context. Only the semantic-search branch of hybrid retrieval and the semantic-dedup check are user-scoped, because those two call sites explicitly pass `user_id` to Qdrant.
- `MemorySystem.clear_memories()` wipes long-term memory **for every user**, and if semantic search is enabled it also drops and recreates the **entire** Qdrant collection (not just the calling user's points). This is why the evaluation harness — which calls `clear_memories()` before every synthetic conversation — must never be run against a shared/production Redis+Qdrant pair.
- `ConsolidationWorker._merge_similar_memories` searches Qdrant for merge candidates without a `user_id` filter, so in a genuinely multi-user deployment it could merge two different users' semantically-similar memories into one record. Decay and promotion, by contrast, operate directly on Redis's `get_recent_memories` (already global) and don't add a *new* cross-user risk beyond what Redis already has.

If real multi-tenant isolation matters for your deployment, the fix is to either namespace Redis keys by `user_id` (e.g. `mem:{user_id}:{memory_id}`, separate type/recency indices per user) or run one Redis DB/keyspace per tenant — this is not something the current code does for you.

---

## 14. Known Bugs & Sharp Edges

These are implementation details worth knowing before you change adjacent code, not exhaustive audit findings:

1. **Access count is incremented twice per retrieved memory, per turn.** `MemoryRetriever._retrieve_with_semantic_search` calls `redis_store.increment_access_count(...)` for every memory in its top-K result (`retriever.py` end of that method), and then `MemorySystem._compose_prompt_context` calls it *again* for the same memories after retrieval returns (`memory_system.py`). Net effect: `access_count` grows twice as fast as intended, which mildly inflates the frequency-ranking signal (weighted only 0.05, so the practical impact is small, but it also means `PROMOTION_ACCESS_THRESHOLD` (5) is reached in roughly half as many actual retrievals as the config value implies).
2. **Docs vs. config drift.** `QUICK_REFERENCE.md` and parts of `README.md` quote `MIN_SEMANTIC_SCORE=0.3`, `MAX_MEMORIES_TO_RETRIEVE=10`, `MEMORY_TOKEN_BUDGET=500`, and a `stats['memories_extracted']` key — none of which match current `config.py` (`0.1`, `50`, `3000`) or `memory_system.py` (`stats['extracted_count']`). Trust the code and this document.
3. **README/QUICK_REFERENCE reference scripts that don't exist in the repo**: `demo.py` (so `make demo` is broken), `test_all_phases.py`, `test_1000_turn_latency.py`, `diagnostic_extraction_phases.py`. Use the scripts that actually exist (listed in the CLAUDE.md "Commands" section) instead.
4. **`get_prompt_context` return type is misdocumented.** Its signature says `-> str` and its docstring says "Returns: Formatted memory context," but it returns the same `(str, List[Dict])` tuple as the private method it forwards to.
5. **Merge search has no user filter** (§13) — a latent cross-user data leak in multi-tenant deployments, currently masked by the fact that Redis itself isn't user-namespaced either.
6. **`detect_update_intent` / `UPDATE_PATTERNS` are dead code** in the live pipeline — defined and exported from config, implemented on `LLMExtractor`, but never called from `extractor.py` or `memory_system.py`. Update detection in practice happens only via the LLM's own `is_update` field in its Stage-3 JSON output.
7. **Promotion doesn't remove the memory from long-term storage.** A promoted memory keeps living in Redis/Qdrant (with `promoted_to_core=True`) *and* gets appended to a core-memory file — so its content can appear twice in a single prompt (once in "CORE MEMORY," once in "LONG-TERM MEMORY (Retrieved)") if it's also still relevant enough to be retrieved.
8. **`EXTRACTION_CLASSIFIER_THRESHOLD` and `HIGH_CONFIDENCE_THRESHOLD` are configured but unused** by any current code path — Stage 2 confidence comes directly from each pattern table's hardcoded value, and core-promotion eligibility uses `PROMOTION_CONFIDENCE_THRESHOLD` instead of `HIGH_CONFIDENCE_THRESHOLD`.
9. **Consolidation is synchronous**, not a background job despite the `ConsolidationWorker` name — a turn that happens to cross the 50-turn interval boundary will block on a full decay+merge+promote pass (each of which does an `O(total_memories)` Redis scan) before `process_turn` returns.

---

## 15. Module Reference Table

| Module | Lines | Responsibility |
|---|---|---|
| `src/config.py` | ~240 | All tunable parameters; loads `.env` |
| `src/flat_file_store.py` | ~190 | Core-memory Markdown files (read/update/append/init templates) |
| `src/redis_store.py` | ~415 | Long-term memory CRUD + 3 Redis indices + confidence boosting/superseding |
| `src/extractor.py` | ~450 | Stage 1 heuristic filter + Stage 2 regex classifier + Stage 2/3 merge logic |
| `src/llm_extractor.py` | ~460 | Stage 3 LLM extraction, multi-provider clients, Groq key rotation, JSON parsing/retry |
| `src/embedding_service.py` | ~180 | sentence-transformers wrapper, singleton model loader, cosine similarity helper |
| `src/vector_store.py` | ~465 | Qdrant collection management, semantic search, dedup search, batch upsert |
| `src/retriever.py` | ~450 | Hybrid + Phase-1 retrieval, 5-signal/3-signal ranking, prompt formatting |
| `src/consolidation_worker.py` | ~520 | Decay, merge, promotion — the Phase 4 background(-in-name) worker |
| `src/memory_system.py` | ~620 | Top-level orchestrator (`process_turn` and the rest of the public API) |
| `evaluation/conversation_generator.py` | ~400 | Synthetic ground-truth conversation generation |
| `evaluation/metrics.py` | ~405 | Extraction/retrieval/distance/consolidation metrics |
| `evaluation/evaluator.py` | ~420 | Evaluation orchestration over `MemorySystem` |
| `run_evaluation.py` | ~155 | CLI entry point for the full Phase 5 suite |

---

*This document was generated from a direct reading of the source in this repository as of the current commit. If you change ranking weights, promotion thresholds, or storage schemas, update the relevant section here — this file, not the marketing-style docs, is meant to track actual behavior.*
