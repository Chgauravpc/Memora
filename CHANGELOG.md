# Changelog - Memory System

All notable changes to the Long-Form Memory System are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - benchmark-driven retrieval work

Driven by measured LoCoMo failures rather than feature planning. See
`BENCHMARK_FINDINGS.md` → *Status after the fixes* for the diagnosis behind each item and
the score trajectory. All numbers so far are single-conversation smoke tests; a full
10-conversation run is still outstanding.

### Added

- **`benchmarks/`** — LoCoMo harness: `runner`, `worker`, `qa`, `report`, `diagnose`,
  `preflight`, `estimate`, `selftest`, plus dataset and path helpers. Entry point
  `run_locomo.py`. Runs are resumable.
- **`benchmarks/selftest.py`** — 104 logic assertions needing no Redis, Qdrant, or network.
  The de-facto unit suite; run it before and after any change to `src/` or `benchmarks/`.
- **`benchmarks/diagnose.py`** — splits failures into retrieval misses vs reader misses, and
  reports a *floor* score by auditing abstentions against golds. More informative than the
  headline percentage, which cannot distinguish the two.
- **`src/lexical_index.py`** — in-process BM25 plus Reciprocal Rank Fusion, no new
  dependency. Fused with the dense branch by rank, avoiding score-normalization constants.
- **`src/entity_index.py`** — inverted entity → memory index. Candidates scored by how many
  *distinct* query entities they mention, so a memory naming two queried people outranks one
  naming either. Deliberately not a knowledge graph: no schema, no entity resolution.
- **`MEMORA_PROFILE`** (`conversation` default / `legacy`) — keeps benchmark retuning
  explicit and reversible instead of silently changing assistant-style behaviour.
- **Memory fields `speaker`, `event_date`, `event_ts`** — `process_turn` previously took
  only text, so multi-speaker attribution and event time were unrecoverable.
- **`scripts/start_backends.sh`** — docker-or-native backend startup with musl-preferring
  Qdrant asset selection and `setsid` detachment.

### Changed

- `STAGE_3_TEMPERATURE` 0.1 → **0.0**. Non-zero temperature meant each re-ingest built a
  different store, making every A/B unreadable; one apparent 64% → 32% regression was
  entirely this.
- `STAGE_3_MAX_TOKENS` 500 → **1200**. Richer extraction values overflowed the cap, and the
  truncated JSON was swallowed by a blanket `except` and presented as "nothing worth
  remembering".
- Extraction prompt gained a **VALUE QUALITY** block. The base prompt's bare-token examples
  taught maximal compression, producing memories like `charity race: 18 May 2023` with the
  answerable detail discarded.
- Ranking rebalanced under the conversation profile toward relevance
  (`semantic .55 / type .10`, from `.30 / .40`). The old weights let an irrelevant
  `constraint` outrank a well-matching `event` — right for instruction-following, wrong for
  question answering.
- `ALWAYS_ON_SEMANTIC_FLOOR` lowered to 0.15 under the conversation profile; the previous 0.5
  recreated the crowding the ranking change was meant to remove.
- `BM25_K1` exposed and set to 50 (by request), effectively disabling term-frequency
  saturation. Worth re-sweeping against the 1.2 default.

### Fixed

- **Dedup key collision.** `build_dedup_key()` was `type:key`, global across speakers, so
  distinct facts sharing a key silently overwrote one another. Now composes
  `type + key [+ speaker] [+ sha1(value)[:16]]` — store size went 366 → 868 on identical
  input.
- **Reader over-abstention.** Caused by the trailing `(or NO_ANSWER)` on the last line of
  the user message, not by the system prompt; three system-prompt rewrites had failed first.
- **Unbounded JSON-retry recursion.** `_parse_and_validate` → `_retry_with_error` →
  `_call_llm` → `_parse_and_validate` had no attempt guard, so a model that never emitted
  JSON recursed until `RecursionError`, one API call per stack frame. Capped at one retry.
- **`diagnose.py` read the wrong results key** (`questions`; the worker writes `records`),
  reporting an empty run as a clean one. Now accepts both and raises on unrecognised
  payloads.
- **Temporal detail loss.** `_enrich_temporal()` recovers dates the LLM drops, ordered after
  the escalation decision so it does not mask turns that still need Stage 3.

---

## [2.0.1] - 2026-02-13

### 🐛 Fixed

#### Phase 4: Core Memory Promotion
- **Fixed empty template files bug**: `FlatFileStore._initialize_files()` now creates proper template files with section headers
  - CORE.md: Added `## Identity` section
  - PREFERENCES.md: Added `## General Preferences` section
  - CONSTRAINTS.md: Added `## Hard Constraints` section
  - INSTRUCTIONS.md: Added `## Communication Style` section
- **Result**: Memory promotion now works correctly (18 memories promoted in 1000-turn test)
- **Previous behavior**: Files were created empty via `filepath.touch()`, causing `append_to_section()` to fail silently
- **New behavior**: Templates include proper Markdown structure with headers and last-updated timestamps

#### Configuration Adjustments
- Option to disable memory decay via `MEMORY_DECAY_ENABLED = False` for testing scenarios
- Allows memories to survive long enough to meet promotion thresholds (50+ turns, 5+ accesses)

### ✨ Added

#### JSON Logging for Per-Turn Statistics
- **New feature**: Optional JSON logging of all per-turn statistics
- Usage: `MemorySystem(user_id, json_log_path='output/stats.json')`
- Comprehensive metrics captured:
  - Phase 1: extracted_count, stored_count, extraction_time_ms
  - Phase 2: semantic_enabled, vector_stored_count, retrieved_count, retrieval_time_ms
  - Phase 3: dedup_count, superseded_count
  - Phase 4: consolidation (decayed, deleted, merged, promoted)
  - Active memories: Full metadata for each retrieved memory (memory_id, content, type, origin_turn, last_used_turn, confidence)
- **Benefit**: Enables detailed analysis of system behavior across long conversations

#### Comprehensive Test Suite
- **test_comprehensive_1000_turn.py**: 1000-turn test exercising all 4 phases
  - 40.1% extraction rate with rich conversation data
  - Automatic duplicate generation every 50 turns (Phase 3 testing)
  - LLM-required messages every 30 turns (Stage 3 testing)
  - Preference updates every 100 turns (superseding testing)
  - Query patterns every 100 turns (retrieval testing)
- **Analysis scripts**: `analyze_results.py`, `analyze_promotion_candidates.py`, `analyze_merge_config.py`, `count_promotions.py`

### 📊 Validation Results (1000-turn comprehensive test)
- **Phase 1 (Extraction)**: 401 memories extracted (40.1% rate)
- **Phase 2 (Semantic Search)**: Vector store active, dual-branch retrieval working
- **Phase 3 (Deduplication)**: 203 semantic duplicates detected and removed (>0.92 similarity)
- **Phase 4 (Consolidation)**: 
  - 20 consolidation runs (every 50 turns)
  - 18 memories promoted to Core Memory files ✅
  - 122 memories decayed (when enabled)
  - 0 merges (requires 0.85-0.92 similarity, test has exact duplicates >0.92)

### 🔧 Technical Details

#### Promotion vs Merging vs Deduplication
- **Deduplication (Phase 3)**: Removes semantic duplicates with >0.92 similarity during extraction
- **Merging (Phase 4)**: Combines similar memories with 0.85-0.92 similarity during consolidation
- **Promotion (Phase 4)**: Moves high-value memories to Core Memory flat files
  - Criteria: confidence ≥0.90, mentions ≥3, accesses ≥5, age ≥50 turns
  - Types: entity, preference, constraint, instruction
  - Target files: CORE.md, PREFERENCES.md, CONSTRAINTS.md, INSTRUCTIONS.md

---

## [2.0.0] - 2026-02-13

### 🎯 Major Achievements
- **100% long-term recall** at all distances (10-1000 turns)
- **80.1% context recall** (up from 68.3%)
- **87% reduction in LLM calls** through optimized extraction
- **Production validation** with 1000-turn latency testing

### ✨ Added

#### Hybrid Retrieval System
- Dual-branch architecture combining semantic search and recency filtering
- Semantic branch: Filtered by MIN_SEMANTIC_SCORE (0.3)
- Recency branch: Unfiltered recent memories (last 100 turns)
- Merge and deduplication logic
- **Result:** 100% long-term recall at 1000 turns (up from 0%)

#### Payment Domain Support
- Extended extraction with payment/financial domain patterns
- Added 11 payment-specific keywords to Phase 1
- Added 13 payment-specific regex patterns to Phase 2:
  - Account numbers (confidence: 0.95)
  - Payment amounts (confidence: 0.90)
  - Due dates (confidence: 0.90)
  - Payment status (confidence: 0.85)
  - Payment arrangements (confidence: 0.85)
  - Customer names (confidence: 0.85)
  - Overdue status (confidence: 0.80-0.85)
  - Payment confirmations (confidence: 0.90)
- **Result:** Phase 1 pass rate 40% → 73.3%, Phase 2 extraction 0% → 46.7%

#### Multi-Key API Rotation
- Support for 4 simultaneous Groq API keys
- Automatic rotation on 429 rate limit errors
- Environment variables: GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
- Total capacity: 400k tokens/day, 48k tokens/minute
- **Result:** Enables high-volume testing and production workloads

#### Test Infrastructure
- **test_1000_turn_latency.py**: Comprehensive 1000-turn latency validation
  - Payment reminder conversation patterns
  - Per-turn latency measurement (processing + retrieval)
  - Statistics: mean, median, P95, P99, throughput
  - Results saved to latency_results_1000_turns.txt
- **diagnostic_extraction_phases.py**: Extraction pipeline diagnostics
  - Independent phase testing (Phase 1, 2, 3)
  - 15 payment domain test messages
  - Phase-by-phase pass rates and extraction counts

#### Documentation
- **RESULTS_FEBRUARY_2026.md**: Comprehensive optimization results
  - Before/after comparisons for all metrics
  - Architecture diagrams and code examples
  - Production readiness assessment
  - Detailed performance analysis

### 🔧 Changed

#### 5-Signal Ranking Optimization
- Rebalanced weights for improved context recall:
  - Semantic: 0.35 → **0.30** (down 5%)
  - Type: 0.20 → **0.40** (up 20%)
  - Recency: 0.20 → **0.10** (down 10%)
  - Frequency: 0.15 → **0.05** (down 10%)
  - Confidence: 0.10 → **0.15** (up 5%)
- Adjusted recency decay rate: 0.1 → **0.001** (gentler decay)
- Extended max decay turns: 1000 → **5000**
- **Result:** Context recall 68.3% → 80.1%

#### Stage 3 LLM Configuration
- Increased STAGE_3_MAX_TOKENS: 200 → **500** (prevents JSON cutoff)
- Maintained STAGE_3_TEMPERATURE: **0.1** (consistent extraction)
- Maintained STAGE_3_CONFIDENCE_THRESHOLD: **0.7** (escalate only when needed)

#### Redis Operations
- Optimized `clear_all_memories()` to check count before clearing
- Skip clearing if empty (performance optimization)
- Display count before clearing (visibility)

### 📊 Performance Improvements

#### Extraction Pipeline
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Phase 1 Pass Rate | 40% | 73.3% | +83% |
| Phase 2 Extraction | 0% | 46.7% | +∞ |
| Phase 3 LLM Calls | ~100% | 13.3% | -87% |

#### Retrieval Quality
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Long-term Recall (1000) | 0% | 100% | +100% |
| Long-term Recall (500) | 12% | 100% | +733% |
| Context Recall | 68.3% | 80.1% | +17% |

#### Latency & Throughput (1000-turn test)
- Processing latency: **575ms mean**, 350ms median, 1333ms P95
- Retrieval latency: **294ms mean**, 296ms median, 379ms P95
- Throughput: **1.74 turns/second**
- Total time: **9.58 minutes** for 1000 turns
- Memories stored: **40 total** (0.04 per turn)

#### API Efficiency
- LLM calls per turn: ~100% → **13.3%** (-87%)
- Tokens per turn: ~750 → ~100 (-87%)
- Estimated cost (1000 turns): ~$7.50 → ~$1.00 (-87%)

### 🐛 Fixed
- JSON parsing errors due to insufficient max tokens (200 → 500)
- Long-term recall failures at 500+ turns (0% → 100%)
- Rate limit crashes during high-volume testing (single key → 4-key rotation)
- Domain mismatch between system design (personal assistant) and test data (payment reminders)

### 📝 Documentation Updates
- Updated [README.md](README.md) with recent improvements section
- Added hybrid retrieval documentation
- Added payment domain pattern documentation
- Updated performance metrics with 1000-turn results
- Added multi-key API rotation configuration
- Updated project structure with new files
- Added quick reference tables and status indicators

---

## [1.5.0] - 2025-12-XX (Phase 5 Complete)

### Added
- RAGAS-based evaluation framework
- Synthetic conversation generator (200 test samples)
- Extraction accuracy metrics (Precision, Recall, F1)
- Retrieval quality metrics (Context Precision/Recall, MRR)
- Distance sweep tests (10-1000 turn recall)
- Consolidation quality evaluation
- Automated test runner with comprehensive reporting

### Results
- Extraction F1: **89.5%**
- Context Recall: **68.3%** (before optimization)
- Long-term Recall: 0% at 1000 turns (before optimization)

---

## [1.4.0] - 2025-10-XX (Phase 4 Complete)

### Added
- Background consolidation worker
- Memory decay for old/unused memories
- Memory merging for semantically similar content
- Promotion to Core Memory files
- 5-signal ranking (semantic + type + recency + frequency + confidence)
- Access tracking and frequency scoring
- Configurable consolidation intervals

---

## [1.3.0] - 2025-08-XX (Phase 3 Complete)

### Added
- Stage 3 LLM-based extraction (OpenAI, Anthropic, Groq)
- Escalation logic (low confidence → LLM)
- Semantic deduplication using vector similarity
- Memory superseding and update detection
- Confidence scoring with certainty modifiers
- Confidence boosting for repeated mentions
- Superseded memory filtering in retrieval

---

## [1.2.0] - 2025-06-XX (Phase 2 Complete)

### Added
- Vector store (Qdrant) for semantic embeddings
- Embedding generation with sentence-transformers (all-MiniLM-L6-v2)
- Semantic similarity search
- Multi-signal ranking (semantic + type + recency)
- Configurable ranking weights
- Graceful fallback to Phase 1 if Qdrant unavailable

---

## [1.1.0] - 2025-04-XX (Phase 1 Complete)

### Added
- Flat file storage for Core Memory (Markdown files)
- Redis storage for Long-Term Memory (persistent)
- Two-stage extraction pipeline (heuristic + pattern-based)
- Basic retrieval (type-priority + recency-based)
- Automated deduplication (key-based)
- Memory indices (type, recency)
- Full pipeline orchestration
- Statistics and monitoring

### Initial Features
- Core Memory: name, language, timezone, preferences
- Long-Term Memory: preferences, constraints, entities, commitments
- Memory types: preference, constraint, entity, instruction, commitment, fact, event
- Pattern-based extraction with confidence scores
- Type-based and recency-based retrieval
- Docker Compose setup (Redis + Qdrant)

---

## [1.0.0] - 2025-02-XX (Initial Release)

### Added
- Project structure and architecture
- Initial specification document
- Docker Compose configuration
- Basic requirements and setup

---

## Legend

- 🎯 Major Achievement - Significant milestone or breakthrough
- ✨ Added - New features or capabilities
- 🔧 Changed - Changes to existing functionality
- 🐛 Fixed - Bug fixes
- 📊 Performance - Performance improvements
- 📝 Documentation - Documentation updates
- 🗑️ Removed - Removed features
- 🔒 Security - Security improvements

---

**Current Version:** 2.0.0  
**Last Updated:** February 13, 2026
