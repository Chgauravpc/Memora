# Memory System - Architecture & Data Flow

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM ARCHITECTURE                       │
│                         (5 Phases Complete)                         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ User Message │  "I prefer calls after 11 AM"
└──────┬───────┘
       │
       ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  PHASE 1-3: EXTRACTION PIPELINE                                  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                   ┃
┃  ┌───────────────────────────────────────────────────────────┐  ┃
┃  │ Stage 1: Sensory Filter (Heuristic)                       │  ┃
┃  │ - Keywords: personal, payment, schedule, etc.             │  ┃
┃  │ - Length, questions, specificity scoring                  │  ┃
┃  │ - Threshold: 0.3 (pass 73.3% of messages)                 │  ┃
┃  │ - Latency: ~1-2ms                                          │  ┃
┃  └────────────────────┬──────────────────────────────────────┘  ┃
┃                       │ PASS                                      ┃
┃                       ▼                                           ┃
┃  ┌───────────────────────────────────────────────────────────┐  ┃
┃  │ Stage 2: Pattern Matching (Regex)                         │  ┃
┃  │ - Personal: name, preferences, location (8 patterns)      │  ┃
┃  │ - Payment: accounts, amounts, dates (13 patterns)         │  ┃
┃  │ - Confidence: 0.8-0.95                                     │  ┃
┃  │ - Extraction rate: 46.7% (high confidence)                │  ┃
┃  │ - Latency: ~1-2ms                                          │  ┃
┃  └────────────────────┬──────────────────────────────────────┘  ┃
┃                       │ confidence < 0.7                          ┃
┃                       ▼                                           ┃
┃  ┌───────────────────────────────────────────────────────────┐  ┃
┃  │ Stage 3: LLM Extraction (Groq)                            │  ┃
┃  │ - Model: llama-3.3-70b-versatile                          │  ┃
┃  │ - Multi-key rotation (4 keys, 400k TPD)                   │  ┃
┃  │ - Structured JSON output                                  │  ┃
┃  │ - Escalation rate: 13.3% (fallback only)                  │  ┃
┃  │ - Latency: 1.1-3.2s per call                              │  ┃
┃  └────────────────────┬──────────────────────────────────────┘  ┃
┃                       │                                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        │
                        ▼
        ┌───────────────────────────┐
        │  Extracted Memory Object  │
        │  {                        │
        │    type: "preference",    │
        │    key: "call_time",      │
        │    value: "after 11 AM",  │
        │    confidence: 0.9        │
        │  }                        │
        └────────────┬──────────────┘
                     │
                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  PHASE 2-3: STORAGE LAYER                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                   ┃
┃  ┌─────────────────────────────────────────────────────────┐    ┃
┃  │ Deduplication Check                                     │    ┃
┃  │ - Key-based: {type}:{key} → memory_id                   │    ┃
┃  │ - Semantic: vector similarity > 0.92                    │    ┃
┃  │ - Confidence boost if repeated mention                  │    ┃
┃  └────────────────────┬────────────────────────────────────┘    ┃
┃                       │ NEW or UPDATE                            ┃
┃                       ▼                                          ┃
┃  ┌──────────────┬──────────────┬───────────────────────────┐   ┃
┃  │              │              │                           │   ┃
┃  │  REDIS       │  QDRANT      │  FLAT FILES               │   ┃
┃  │  (Primary)   │  (Vectors)   │  (Core Memory)            │   ┃
┃  │              │              │                           │   ┃
┃  │  Stores:     │  Stores:     │  Always-injected:         │   ┃
┃  │  - Memories  │  - Embeddings│  - Identity (name, lang)  │   ┃
┃  │  - Metadata  │  - 384-dim   │  - Preferences            │   ┃
┃  │  - Indices   │  - Fast      │  - Instructions           │   ┃
┃  │              │    search    │  - Constraints            │   ┃
┃  │              │              │                           │   ┃
┃  │  AOF persist │  Snapshots   │  Human-editable .md       │   ┃
┃  │              │              │                           │   ┃
┃  └──────────────┴──────────────┴───────────────────────────┘   ┃
┃                                                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        │
                        │ (Later, on query)
                        ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  PHASE 5: HYBRID RETRIEVAL SYSTEM (NEW)                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                   ┃
┃  Query: "What time should I call?"                               ┃
┃                                                                   ┃
┃  ┌─────────────────────────┐  ┌──────────────────────────────┐ ┃
┃  │  BRANCH 1: Semantic     │  │  BRANCH 2: Recency           │ ┃
┃  │                         │  │                              │ ┃
┃  │  Qdrant vector search   │  │  Redis sorted set by turn    │ ┃
┃  │  Filter: sim > 0.3      │  │  NO similarity filter        │ ┃
┃  │  Returns: top 20        │  │  Returns: last 100 turns     │ ┃
┃  │  Latency: 24-35ms       │  │  Latency: <10ms              │ ┃
┃  │                         │  │                              │ ┃
┃  └────────────┬────────────┘  └─────────────┬────────────────┘ ┃
┃               │                              │                   ┃
┃               └──────────┬───────────────────┘                   ┃
┃                          ▼                                       ┃
┃           ┌──────────────────────────────┐                      ┃
┃           │  MERGE & DEDUPLICATE         │                      ┃
┃           │  - Combine both branches     │                      ┃
┃           │  - Remove duplicates by ID   │                      ┃
┃           └──────────────┬───────────────┘                      ┃
┃                          ▼                                       ┃
┃           ┌──────────────────────────────┐                      ┃
┃           │  5-SIGNAL RANKING            │                      ┃
┃           │                              │                      ┃
┃           │  final_score = Σ(w_i × s_i) │                      ┃
┃           │                              │                      ┃
┃           │  - Semantic:   30% (content) │                      ┃
┃           │  - Type:       40% (priority)│                      ┃
┃           │  - Recency:    10% (time)    │                      ┃
┃           │  - Frequency:   5% (access)  │                      ┃
┃           │  - Confidence: 15% (quality) │                      ┃
┃           │                              │                      ┃
┃           └──────────────┬───────────────┘                      ┃
┃                          ▼                                       ┃
┃           ┌──────────────────────────────┐                      ┃
┃           │  TOP K MEMORIES (K=10)       │                      ┃
┃           └──────────────────────────────┘                      ┃
┃                                                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                        │
                        ▼
            ┌───────────────────────┐
            │  Memory Context       │
            │                       │
            │  "User prefers calls  │
            │   after 11 AM          │
            │   (mentioned turn 42,  │
            │    confidence: 0.9)"   │
            └────────────┬──────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  LLM Prompt      │
              │  + Context       │
              │  + User Message  │
              └──────────────────┘
```

---

## Data Flow: Process Turn

```
┌──────────────────────────────────────────────────────────────────┐
│  memory.process_turn("I prefer calls after 11 AM")              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────┐                   ┌──────────────────┐
│  1. EXTRACT     │                   │  2. RETRIEVE     │
│                 │                   │                  │
│  3-stage:       │                   │  Hybrid:         │
│  - Sensory      │                   │  - Semantic      │
│  - Pattern      │                   │  - Recency       │
│  - LLM (13.3%)  │                   │  - 5-signal      │
│                 │                   │                  │
│  Result:        │                   │  Result:         │
│  [{type: "pref",│                   │  [mem1, mem2,    │
│    key: "call", │                   │   mem3, ...]     │
│    value: "11", │                   │                  │
│    conf: 0.9}]  │                   │  Latency: 294ms  │
│                 │                   │                  │
│  Latency: 575ms │                   │                  │
└────────┬────────┘                   └────────┬─────────┘
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│  3. STORE       │                            │
│                 │                            │
│  - Dedup check  │                            │
│  - Redis insert │                            │
│  - Vector index │                            │
│  - Update stats │                            │
│                 │                            │
│  Latency: 130ms │                            │
└────────┬────────┘                            │
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │  4. FORMAT CONTEXT     │
           │                        │
           │  Core Memory (500 tok) │
           │  + Long-term (500 tok) │
           │  = 1000 tokens total   │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │  RETURN                │
           │                        │
           │  (context, stats)      │
           │                        │
           │  stats = {             │
           │    extracted: 1,       │
           │    retrieved: 5,       │
           │    active: [...],      │
           │    latency: {...}      │
           │  }                     │
           └────────────────────────┘
```

---

## Performance Optimization Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  PROBLEM: 0% recall at 1000 turns                                │
│  CAUSE: Single-branch semantic search with tight filter          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  SOLUTION 1: Hybrid Retrieval (Dual-Branch)                      │
│                                                                   │
│  BEFORE:                          AFTER:                         │
│  ┌────────────────┐               ┌────────────────┐            │
│  │ Semantic Only  │               │ Semantic Branch│            │
│  │ Filter: > 0.3  │               │ + Recency Branch│           │
│  │ Result: 0%     │               │ Result: 100%   │            │
│  │ at 1000 turns  │               │ at 1000 turns  │            │
│  └────────────────┘               └────────────────┘            │
│                                                                   │
│  IMPACT: +100% long-term recall                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PROBLEM: 100% LLM calls, rate limit crashes                     │
│  CAUSE: Pattern matching failed on payment domain                │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  SOLUTION 2: Payment Domain Patterns                             │
│                                                                   │
│  ADDED:                          RESULT:                         │
│  - 11 payment keywords           Phase 1: 40% → 73.3%            │
│  - 13 payment patterns           Phase 2: 0% → 46.7%             │
│                                  Phase 3: 100% → 13.3%           │
│                                                                   │
│  EXAMPLES:                                                        │
│  "account ending in 4567"  → Pattern match (0ms, $0)             │
│  "payment of $450"         → Pattern match (0ms, $0)             │
│  "due on February 5th"     → Pattern match (0ms, $0)             │
│                                                                   │
│  IMPACT: 87% cost reduction, 87% latency reduction               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PROBLEM: Rate limits with single API key                        │
│  CAUSE: 100k tokens/day limit insufficient for 1000 turns        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  SOLUTION 3: Multi-Key API Rotation                              │
│                                                                   │
│  BEFORE:                          AFTER:                         │
│  1 key × 100k TPD = 100k          4 keys × 100k TPD = 400k      │
│  Result: Crash at turn 10         Result: Smooth 1000 turns     │
│                                                                   │
│  ROTATION:                                                        │
│  Key 1 → Rate limit (429) → Key 2 → Rate limit → Key 3 → ...   │
│                                                                   │
│  IMPACT: 4x capacity, no crashes                                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PROBLEM: Context recall 68.3% (not enough relevant context)     │
│  CAUSE: Poor ranking weight balance                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  SOLUTION 4: 5-Signal Ranking Optimization                       │
│                                                                   │
│  WEIGHTS REBALANCED:                                             │
│  Signal      │ Before │ After │ Rationale                        │
│  ───────────┼────────┼───────┼─────────────────────────────     │
│  Semantic   │  35%   │  30%  │ ↓ Content less critical          │
│  Type       │  20%   │  40%  │ ↑ Constraints/instructions key   │
│  Recency    │  20%   │  10%  │ ↓ Hybrid branch handles this     │
│  Frequency  │  15%   │   5%  │ ↓ Secondary signal               │
│  Confidence │  10%   │  15%  │ ↑ Quality matters                │
│                                                                   │
│  RESULT: Context recall 68.3% → 80.1% (+17%)                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Consolidation Pipeline (Phase 4)

```
┌──────────────────────────────────────────────────────────────────┐
│  TRIGGER: Every 50 turns                                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
           ┌─────────────────────────────────┐
           │  BACKGROUND CONSOLIDATION       │
           └────────────┬────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   DECAY      │ │    MERGE     │ │   PROMOTE    │
│              │ │              │ │              │
│ Old/unused   │ │ Similar      │ │ Important    │
│ memories     │ │ memories     │ │ → Core Mem   │
│              │ │              │ │              │
│ If:          │ │ If:          │ │ If:          │
│ - age > 500  │ │ - sim > 0.85 │ │ - conf ≥ 0.90│
│ - unused     │ │ - same type  │ │ - mention ≥ 3│
│   > 300 turn │ │              │ │ - access ≥ 5 │
│              │ │              │ │ - age ≥ 50   │
│ Then:        │ │ Then:        │ │ - promotable │
│ conf × 0.995 │ │ Combine +    │ │   type       │
│              │ │ boost conf   │ │              │
│              │ │              │ │ Then:        │
│              │ │              │ │ Append to    │
│              │ │              │ │ CORE.md/etc  │
└──────────────┘ └──────────────┘ └──────────────┘

Note: Promotion writes to flat files with proper section headers:
- CORE.md → ## Identity (entities)
- PREFERENCES.md → ## General Preferences (preferences)
- CONSTRAINTS.md → ## Hard Constraints (constraints)
- INSTRUCTIONS.md → ## Communication Style (instructions)
```

---

## Latency Breakdown (1000-Turn Test)

```
┌────────────────────────────────────────────────────────────────┐
│  PROCESSING LATENCY (per turn)                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ▓▓▓▓▓▓▓▓░░░░░░  Stage 1: Sensory (1-2ms)                      │
│  ▓▓▓▓▓▓▓▓░░░░░░  Stage 2: Pattern (1-2ms)                      │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Stage 3: LLM (1100-3200ms)    │
│  ▓▓▓▓▓▓▓▓▓▓▓▓░░  Storage (130ms)                               │
│                                                                 │
│  MEAN: 575ms                                                    │
│  - Without LLM (87% of turns): ~135ms                          │
│  - With LLM (13% of turns): ~1200-3300ms                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  RETRIEVAL LATENCY (per turn)                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Semantic branch (24-35ms)                     │
│  ▓▓▓▓░░░░░░░░░░  Recency branch (<10ms)                        │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  5-signal ranking (150-200ms)  │
│  ▓▓▓▓▓▓░░░░░░░░  Formatting (50-80ms)                          │
│                                                                 │
│  MEAN: 294ms (very consistent, σ=68ms)                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Recall Performance Across Distances

```
┌────────────────────────────────────────────────────────────────┐
│  LONG-TERM RECALL (Distance Sweep)                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  100% ┼─────────────────────────────────────────────          │
│       │ ████████████████████████████████████████ 100%         │
│       │                                                         │
│   90% ┤                                                         │
│       │                                                         │
│   80% ┤                                                         │
│       │                                                         │
│   70% ┤                                                         │
│       │                                                         │
│   60% ┤                                                         │
│       │                                                         │
│   50% ┤                                                         │
│       │                                                         │
│   40% ┤                                                         │
│       │                                                         │
│   30% ┤                                                         │
│       │                                                         │
│   20% ┤                                                         │
│       │                                                         │
│   10% ┤                                                         │
│       │                                                         │
│    0% ┼────┬────┬────┬────┬────                                │
│          10   50  100  500 1000  (turns ago)                   │
│                                                                 │
│  ✅ 100% recall at ALL distances                                │
└────────────────────────────────────────────────────────────────┘
```

---

## System Status Dashboard

```
╔═══════════════════════════════════════════════════════════════╗
║            MEMORY SYSTEM STATUS DASHBOARD                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  📊 PERFORMANCE                                               ║
║  ├─ Processing Latency:    575ms mean     ✅                 ║
║  ├─ Retrieval Latency:     294ms mean     ✅                 ║
║  ├─ Throughput:            1.74 turns/sec ✅                 ║
║  └─ Stability:             1000+ turns    ✅                 ║
║                                                               ║
║  🎯 QUALITY                                                   ║
║  ├─ Long-term Recall:      100% (1000)    ✅                 ║
║  ├─ Context Recall:        80.1%          ✅                 ║
║  ├─ Extraction F1:         89.5%          ✅                 ║
║  └─ Pattern Coverage:      87%            ✅                 ║
║                                                               ║
║  💰 EFFICIENCY                                                ║
║  ├─ LLM Calls:             13.3% of turns ✅                 ║
║  ├─ Pattern-based:         87% cost-free  ✅                 ║
║  ├─ API Capacity:          400k tokens/day✅                 ║
║  └─ Cost per 1000 turns:   ~$1.00         ✅                 ║
║                                                               ║
║  🔧 INFRASTRUCTURE                                            ║
║  ├─ Redis:                 AOF persistent ✅                 ║
║  ├─ Qdrant:                Vector indexed ✅                 ║
║  ├─ API Keys:              4-key rotation ✅                 ║
║  └─ Monitoring:            Ready for setup📋                 ║
║                                                               ║
║  STATUS: ✅ PRODUCTION-READY                                  ║
║  VERSION: 2.0.0                                               ║
║  LAST UPDATED: February 13, 2026                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**For complete documentation, see [README.md](README.md)**  
**For performance details, see [RESULTS_FEBRUARY_2026.md](RESULTS_FEBRUARY_2026.md)**
