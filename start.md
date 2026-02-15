# Getting Started - Memory System Setup Guide

This guide walks you through setting up and running the Long-Form Memory System on a new computer from scratch.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python --version`

2. **Docker Desktop**
   - Windows/Mac: https://www.docker.com/products/docker-desktop
   - Linux: https://docs.docker.com/engine/install/
   - Verify installation: `docker --version` and `docker-compose --version`

3. **Git** (optional, for cloning)
   - Download from: https://git-scm.com/downloads
   - Verify installation: `git --version`

### API Keys

You'll need at least one API key from a supported LLM provider:

- **Groq** (Recommended - fastest, free tier: 100k tokens/day)
  - Sign up at: https://console.groq.com/
  - Get API key from: https://console.groq.com/keys
  
- **OpenAI** (Alternative)
  - Sign up at: https://platform.openai.com/
  - Get API key from: https://platform.openai.com/api-keys
  
- **Anthropic** (Alternative)
  - Sign up at: https://console.anthropic.com/
  - Get API key from: https://console.anthropic.com/settings/keys

---

## 🚀 Installation Steps

### Step 1: Get the Code

Navigate to the project directory:

```bash
cd memory-system-phase1
```

Or if cloning from a repository:

```bash
git clone <repository-url>
cd memory-system-phase1
```

### Step 2: Install Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- redis (Redis client)
- qdrant-client (Vector database)
- sentence-transformers (Embeddings)
- groq, openai, anthropic (LLM providers)

For evaluation features (optional):

```bash
pip install -r requirements_evaluation.txt
```

### Step 3: Configure Environment Variables

1. Copy the example environment file:

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

2. Edit the `.env` file with your API key(s):

```bash
# Single API key (minimum required)
GROQ_API_KEY=your_groq_api_key_here

# OR for OpenAI
OPENAI_API_KEY=your_openai_key_here

# OR for Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**For Production (Recommended):** Use multiple Groq keys for rate limit rotation:

```bash
GROQ_API_KEY=your_first_key
GROQ_API_KEY_1=your_second_key
GROQ_API_KEY_2=your_third_key
GROQ_API_KEY_3=your_fourth_key
```

This gives you 400k tokens/day total capacity with automatic failover.

### Step 4: Start Backend Services

Start Redis and Qdrant using Docker Compose:

```bash
docker-compose up -d
```

This command will:
- Start Redis on port 6379 (long-term memory storage)
- Start Qdrant on port 6333 (vector database for semantic search)
- Run both services in the background (`-d` flag)

Verify services are running:

```bash
docker-compose ps
```

You should see both `redis` and `qdrant` with status "Up".

---

## ✅ Verification

### Check Service Health

Run this quick health check:

```bash
python -c "from src import MemorySystem; m = MemorySystem('test'); print(m.health_check())"
```

Expected output:

```python
{
  'redis': True,
  'flat_files': True,
  'vector_store': True
}
```

If any service shows `False`, check:
- Docker containers are running: `docker-compose ps`
- Ports are not in use: 6379 (Redis), 6333 (Qdrant)
- Firewall isn't blocking connections

---

## 🎮 Running the System

### Run the Comprehensive 1000-Turn Test

Run the full production validation test:

```bash
python test_comprehensive_1000_turn.py
```

This validates:
- All 4 phases working together
- Extraction, deduplication, consolidation, promotion
- Creates JSON output with detailed statistics
- Generates 53 promoted memories to Core Memory files

**Note:** This test takes ~5 minutes and makes ~130 LLM API calls.

---

## 📊 Understanding the Output

### Memory Files

After running the test, check the `memory/` directory:

```bash
memory/
└── comprehensive_test/        # From 1000-turn test
    ├── CORE.md               # Core identity (always injected)
    ├── PREFERENCES.md        # User preferences (24 promoted)
    ├── CONSTRAINTS.md        # Hard constraints (6 promoted)
    └── INSTRUCTIONS.md       # Communication style (6 promoted)
```

These are human-readable Markdown files you can edit directly. The test promotes 53 high-value memories to these files.

### JSON Statistics

The comprehensive test creates detailed statistics:

```bash
output/
└── comprehensive_1000_turn_stats.json  # Per-turn metrics
```

This file contains:
- Extraction counts per turn
- Retrieval performance
- Active memories with full metadata
- Consolidation statistics

---

## 🔧 Configuration

### Key Configuration Files

1. **src/config.py** - Main configuration
   - Extraction thresholds
   - Ranking weights
   - Decay/merge/promotion settings
   - Multi-key setup

2. **.env** - Environment variables
   - API keys
   - Service endpoints

3. **docker-compose.yml** - Service configuration
   - Redis persistence (AOF)
   - Qdrant storage location

### Common Adjustments

#### Change LLM Provider

Edit `src/config.py`:

```python
LLM_PROVIDER = "groq"  # "openai" | "anthropic" | "groq"
```

#### Adjust Extraction Rate

Edit `src/config.py`:

```python
SENSORY_FILTER_THRESHOLD = 0.3  # Lower = more extraction
STAGE_3_CONFIDENCE_THRESHOLD = 0.7  # Lower = more LLM calls
```

#### Modify Ranking Weights

Edit `src/config.py`:

```python
RANKING_WEIGHTS_5_SIGNAL = {
    "semantic": 0.30,    # Content relevance
    "type": 0.40,        # Type priority
    "recency": 0.10,     # Time decay
    "frequency": 0.05,   # Access count
    "confidence": 0.15,  # Quality score
}
```

#### Enable/Disable Features

Edit `src/config.py`:

```python
SEMANTIC_SEARCH_ENABLED = True       # Vector search
SEMANTIC_DEDUP_ENABLED = True        # Duplicate detection
CONSOLIDATION_ENABLED = True         # Background consolidation
MEMORY_DECAY_ENABLED = False         # Memory decay (disabled for testing)
PROMOTION_ENABLED = True             # Core Memory promotion
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused" to Redis

**Solution:**
```bash
# Check if Docker is running
docker ps

# Restart services
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs redis
```

### Issue: "Rate limit exceeded" from Groq

**Solution:**
- Add more API keys to `.env` (see Step 3)
- Or wait for rate limit to reset (daily/minute limits)
- Or reduce extraction rate in `config.py`

### Issue: "sentence-transformers model not found"

**Solution:**
```bash
# First run downloads the model (~90MB)
# Ensure internet connection is available
# Or pre-download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: High memory usage

**Solution:**
- Qdrant stores vectors in memory by default
- For large datasets (>10k memories), configure disk storage in `docker-compose.yml`
- Or clear old memories: `python -c "from src import MemorySystem; m = MemorySystem('user'); m.clear_memories()"`

### Issue: Slow extraction

**Solution:**
- LLM calls (Stage 3) take 1-3 seconds
- Reduce Stage 3 usage by improving Phase 1/2 patterns in `extractor.py`
- Or use Groq (fastest provider) instead of OpenAI/Anthropic

---

## 📈 Next Steps

### Verify Results

1. **Check promoted memories:**
   ```bash
   # View the promoted Core Memory files
   cat memory/comprehensive_test/PREFERENCES.md
   cat memory/comprehensive_test/CONSTRAINTS.md
   ```

2. **Review test statistics:**
   - View `output/comprehensive_1000_turn_stats.json` for per-turn metrics
   - Expected: 401 memories extracted, 53 promoted, 203 duplicates removed

3. **Check production metrics:**
   - View `RESULTS_FEBRUARY_2026.md` for benchmark results
   - Expected: 100% long-term recall, 80.1% context recall, <1s latency

### Integrate with Your Application

```python
from src import MemorySystem

# Initialize for a user
memory = MemorySystem(user_id="alice", json_log_path="logs/alice.json")

# Process each conversation turn
context, stats = memory.process_turn(
    user_message="I prefer Python for backend work",
    priority_types=["constraint", "preference"]
)

# Use context in your LLM prompt
prompt = f"""
{context}

User: {user_message}
Assistant: [Your response here]
"""

# Access active memories
for mem in stats['active_memories']:
    print(f"{mem['type']}: {mem['value']} (confidence: {mem['confidence']})")
```

### Monitor Performance

Enable JSON logging to track system behavior:

```python
memory = MemorySystem(
    user_id="alice",
    json_log_path="output/alice_stats.json"
)
```

Analyze the JSON file to monitor:
- Extraction rates per turn
- Retrieval latency
- Memory growth
- Consolidation impact

---

## 📚 Additional Resources

- **README.md** - Complete documentation
- **ARCHITECTURE.md** - System design and data flow
- **QUICK_REFERENCE.md** - Developer cheat sheet
- **RESULTS_FEBRUARY_2026.md** - Performance benchmarks
- **CHANGELOG.md** - Version history

---

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review service logs: `docker-compose logs`
3. Verify configuration in `src/config.py`
4. Check API key validity: test with direct API call
5. Ensure all dependencies installed: `pip list`

---

## ✨ Summary Checklist

- [ ] Python 3.8+ installed
- [ ] Docker Desktop installed and running
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with API key(s)
- [ ] Services started (`docker-compose up -d`)
- [ ] Health check passed
- [ ] 1000-turn test completed successfully

**You're ready to use the Memory System! 🎉**

---

**Quick Start Command Sequence:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Start services
docker-compose up -d

# 4. Run comprehensive test
python test_comprehensive_1000_turn.py
```

That's it! The system is now running and ready for integration.
