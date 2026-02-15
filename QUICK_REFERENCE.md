# Memory System - Quick Reference

## 🚀 Quick Start

```bash
# 1. Setup
pip install -r requirements.txt
docker-compose up -d

# 2. Configure API Key
export GROQ_API_KEY=your_key_here

# 3. Run Demo
python demo_phase4.py
```

---

## 📊 Key Metrics (Production Validated)

| Metric | Value |
|--------|-------|
| **Long-term Recall (1000 turns)** | 100% ✅ |
| **Context Recall** | 80.1% ✅ |
| **Extraction F1** | 89.5% ✅ |
| **Processing Latency (mean)** | 575ms ✅ |
| **Retrieval Latency (mean)** | 294ms ✅ |
| **Throughput** | 1.74 turns/sec ✅ |
| **LLM Calls** | 13.3% (87% pattern-based) ✅ |

---

## 🔧 Configuration Quick Reference

### Environment Variables (.env)
```bash
# Single key
GROQ_API_KEY=your_key_here

# Multi-key rotation (recommended for production)
GROQ_API_KEY=key_1
GROQ_API_KEY_1=key_2
GROQ_API_KEY_2=key_3
GROQ_API_KEY_3=key_4  # 400k tokens/day total
```

### Key Parameters (src/config.py)

#### Extraction Thresholds
```python
SENSORY_FILTER_THRESHOLD = 0.3      # Phase 1 pass threshold
EXTRACTION_CLASSIFIER_THRESHOLD = 0.6  # Phase 2 confidence
STAGE_3_CONFIDENCE_THRESHOLD = 0.7  # Escalate to LLM if below
```

#### Retrieval Settings
```python
# Hybrid Retrieval
HYBRID_RETRIEVAL_ENABLED = True
MIN_SEMANTIC_SCORE = 0.3            # Semantic branch filter
MAX_MEMORIES_TO_RETRIEVE = 10       # Top K memories
MEMORY_TOKEN_BUDGET = 500           # Max tokens for context

# 5-Signal Ranking Weights (Optimized)
semantic:   0.30  # Content relevance
type:       0.40  # Type priority (constraints > instructions)
recency:    0.10  # Time decay (gentle: 0.001 rate)
frequency:  0.05  # Access count
confidence: 0.15  # Memory confidence
```

#### LLM Settings
```python
LLM_PROVIDER = "groq"
LLM_EXTRACTION_MODEL = "llama-3.3-70b-versatile"
STAGE_3_MAX_TOKENS = 500            # Prevent JSON cutoff
STAGE_3_TEMPERATURE = 0.1           # Consistent extraction
```

---

## 📝 Basic Usage

### Initialize Memory System
```python
from src import MemorySystem

memory = MemorySystem(user_id="alice")
```

### Process Conversation Turn
```python
# Process turn (extract + store + retrieve)
memory_context, stats = memory.process_turn("I prefer calls after 11 AM")

# Access statistics
print(f"Extracted: {stats['memories_extracted']}")
print(f"Retrieved: {stats['memories_retrieved']}")
print(f"Active: {len(stats['active_memories'])}")
```

### Retrieve Only (No Processing)
```python
# Get memory context without processing
context = memory.get_prompt_context(
    user_message="What time should I call?",
    priority_types=["preference", "constraint"]
)
```

### Update Core Memory
```python
# Update always-injected identity
memory.update_core_memory(
    file="CORE.md",
    section="Identity",
    field="Name",
    value="Alice"
)
```

### Get Statistics
```python
stats = memory.get_statistics()
print(f"Total memories: {stats['total_memories']}")
print(f"By type: {stats['memories_by_type']}")
```

### JSON Logging (Per-Turn Stats)
```python
# Enable comprehensive logging
memory = MemorySystem(
    user_id="alice",
    json_log_path="output/conversation_stats.json"
)

# Stats are automatically saved after each turn
_, stats = memory.process_turn("I prefer Python")

# Access in-memory log
log = memory.get_turn_stats_log()

# Export manually
memory.export_turn_stats("output/manual_export.json")
```

**Captured Metrics:**
- Phase 1: extracted_count, stored_count
- Phase 2: semantic_enabled, vector_stored_count, retrieved_count
- Phase 3: dedup_count, superseded_count
- Phase 4: consolidation (decayed, merged, promoted)
- Active memories with full metadata

### Clear Memories
```python
# Clear all memories for user
memory.clear_memories()
```

---

## 🧪 Testing Commands

### Quick Tests
```bash
# Phase 4 demo (consolidation)
python demo_phase4.py

# 60-turn customer service
python test_customer_conversation.py

# Active memory tracking
python demo_active_memories.py
```

### Comprehensive Tests
```bash
# 120+ turn full system test
python test_all_phases.py

# 1000-turn latency validation (REQUIRES API KEYS)
python test_1000_turn_latency.py

# Extraction diagnostics
python diagnostic_extraction_phases.py
```

### Evaluation
```bash
# RAGAS evaluation (200 conversations)
pip install -r requirements_evaluation.txt
python run_evaluation.py
```

---

## 🔍 Troubleshooting

### Redis Connection Error
```bash
docker-compose ps  # Check if redis is running
docker-compose up -d redis  # Start redis
```

### Qdrant Connection Error
```bash
docker-compose ps  # Check if qdrant is running
docker-compose up -d qdrant  # Start qdrant
# Note: System falls back to Phase 1 if Qdrant unavailable
```

### Rate Limit Errors
```bash
# Add more API keys to .env
export GROQ_API_KEY_1=second_key
export GROQ_API_KEY_2=third_key
export GROQ_API_KEY_3=fourth_key
```

### No Memories Extracted
1. Check extraction thresholds in config.py
2. Run diagnostic_extraction_phases.py
3. Enable DEBUG logging

### Slow First Query
- Normal: Embedding model loads (~16s cold start)
- Subsequent queries are fast (24-52ms)

---

## 📁 Project Structure

```
memory-system-phase1/
├── src/                  # Source code
│   ├── config.py         # ⚙️ Configuration (tune here)
│   ├── extractor.py      # 🔍 3-stage extraction
│   ├── retriever.py      # 🎯 Hybrid retrieval
│   ├── memory_system.py  # 🧠 Main orchestrator
│   └── ...
├── memory/               # 💾 Flat file storage
│   └── user_id/
│       ├── CORE.md       # Always-injected identity
│       ├── PREFERENCES.md
│       ├── INSTRUCTIONS.md
│       └── CONSTRAINTS.md
├── README.md             # 📖 Complete documentation
├── RESULTS_FEBRUARY_2026.md  # 📊 Benchmarks ⭐
├── CHANGELOG.md          # 📝 Version history
└── docker-compose.yml    # 🐳 Redis + Qdrant
```

---

## 🎯 Memory Types

| Type | Priority | Use Case | Example |
|------|----------|----------|---------|
| **constraint** | 1.0 | Hard rules | "Never share my email" |
| **instruction** | 0.95 | Behavioral | "Address me as Dr. Smith" |
| **commitment** | 0.8 | Time-bound | "Payment due Feb 15th" |
| **preference** | 0.7 | User likes | "Prefer dark mode" |
| **entity** | 0.6 | People/places | "Manager: John Doe" |
| **fact** | 0.5 | General info | "Account balance: $450" |
| **event** | 0.4 | Past actions | "Met yesterday" |

---

## 📊 Extraction Pipeline

```
User Message
     ↓
┌─────────────────────┐
│  PHASE 1: Sensory   │  73.3% pass rate
│  Keyword + patterns │  Fast: ~1-2ms
└──────────┬──────────┘
           ↓ (pass)
┌─────────────────────┐
│  PHASE 2: Patterns  │  46.7% extraction
│  13 payment patterns│  Fast: ~1-2ms
│  8 personal patterns│  Confidence: 0.8-0.95
└──────────┬──────────┘
           ↓ (confidence < 0.7)
┌─────────────────────┐
│  PHASE 3: LLM       │  13.3% escalation
│  Groq API           │  Slow: 1.1-3.2s
│  4-key rotation     │  High confidence
└─────────────────────┘

Result: 87% fast (pattern), 13% slow (LLM)
```

---

## 🔄 Retrieval Pipeline

```
Query Message
     ↓
┌──────────────────────────────┐
│   HYBRID RETRIEVAL (Phase 5) │
├──────────────────────────────┤
│  Branch 1: Semantic Search   │  Filter: sim > 0.3
│  - Qdrant vector search      │  Returns: top 20
│  - Content relevance         │
├──────────────────────────────┤
│  Branch 2: Recency Search    │  No filter
│  - Redis sorted set          │  Returns: last 100
│  - Recent context            │
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│  MERGE & DEDUPLICATE         │  Remove duplicates
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│  5-SIGNAL RANKING            │
│  semantic × 0.30             │
│  + type × 0.40               │  Rebalanced weights
│  + recency × 0.10            │  (February 2026)
│  + frequency × 0.05          │
│  + confidence × 0.15         │
└──────────┬───────────────────┘
           ↓
    Top K Memories
```

---

## 🚀 Performance Tips

### For High-Volume Production
1. ✅ Use 4 API keys (400k tokens/day)
2. ✅ Enable hybrid retrieval (100% recall)
3. ✅ Optimize Phase 1/2 patterns for your domain
4. ✅ Set up Redis AOF persistence
5. ✅ Monitor latency and recall metrics

### For Cost Optimization
1. ✅ Add domain-specific patterns (reduce LLM calls)
2. ✅ Tune STAGE_3_CONFIDENCE_THRESHOLD (higher = fewer LLM calls)
3. ✅ Enable semantic deduplication (prevent bloat)
4. ✅ Use consolidation (decay old memories)

### For Best Recall
1. ✅ Enable hybrid retrieval
2. ✅ Lower MIN_SEMANTIC_SCORE if needed (0.2-0.3)
3. ✅ Increase MAX_MEMORIES_TO_RETRIEVE if needed
4. ✅ Tune 5-signal weights for your use case

---

## 📞 Support

### Documentation
- **Complete Guide:** [README.md](README.md)
- **Performance:** [RESULTS_FEBRUARY_2026.md](RESULTS_FEBRUARY_2026.md)
- **Changes:** [CHANGELOG.md](CHANGELOG.md)
- **Overview:** [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)

### Common Issues
- Troubleshooting section in README.md
- diagnostic_extraction_phases.py for diagnostics
- Enable DEBUG logging for detailed output

---

## ✅ Production Checklist

Before deploying to production:

- [ ] Multi-key API rotation configured (4 keys)
- [ ] Redis AOF persistence enabled
- [ ] Qdrant backups configured
- [ ] Monitoring set up (Prometheus/Grafana)
- [ ] Alerts configured (latency, recall, errors)
- [ ] Load testing completed (1000+ turns)
- [ ] Domain-specific patterns added
- [ ] Security review completed
- [ ] Backup/restore procedure tested
- [ ] Documentation updated for your domain

---

**Version:** 2.0.0  
**Status:** ✅ Production-Ready  
**Last Updated:** February 13, 2026

**For complete documentation, see [README.md](README.md)**
