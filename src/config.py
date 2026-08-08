"""
Configuration for the Memory System - Phase 1, 2 & 3
All tunable parameters from Section 12 of the spec
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Qdrant Configuration (Phase 2)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memory_vectors")

# Embedding Configuration (Phase 2)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Memory Layer Configuration
CORE_MEMORY_FILES = ["CORE.md", "PREFERENCES.md", "INSTRUCTIONS.md", "CONSTRAINTS.md"]
CORE_MEMORY_TOKEN_BUDGET = 500  # Always injected

# Extraction Configuration (Phase 1: Stage 1 & 2 only)
SENSORY_FILTER_THRESHOLD = 0.3  # Heuristic score threshold
EXTRACTION_CLASSIFIER_THRESHOLD = 0.6  # Classifier confidence threshold

# Heuristic weights for sensory filter
HEURISTIC_WEIGHTS = {
    "length": 0.3,  # Longer messages more likely to contain info
    "keywords": 0.4,  # Presence of important keywords
    "question": 0.15,  # Questions often contain context
    "specificity": 0.15,  # Specific details vs vague statements
}

# Keywords that signal extractable information
EXTRACTION_KEYWORDS = {
    "preference": ["prefer", "like", "hate", "love", "favorite", "always", "never"],
    "constraint": ["must", "cannot", "don't", "won't", "shouldn't", "allergic", "avoid"],
    "entity": ["my", "named", "called", "manager", "friend", "colleague", "family"],
    "instruction": ["always", "whenever", "remember to", "make sure", "don't forget"],
    "commitment": ["will", "promise", "committed", "deadline", "by", "before"],
    "fact": ["live in", "work at", "am", "is", "from", "born", "studied"],
    "payment": ["payment", "account", "balance", "due", "amount", "bill", "paid", "extension", "plan", "outstanding", "received"],
}

# Memory Types (for Redis indexing)
MEMORY_TYPES = ["preference", "constraint", "entity", "instruction", "commitment", "fact", "event"]

# Retrieval Configuration
MAX_MEMORIES_TO_RETRIEVE = 50  # Top K memories to inject (increased for better long-term recall)
MEMORY_TOKEN_BUDGET = 3000  # Total token budget for retrieved memories

# Phase 2: Semantic Search Configuration
SEMANTIC_SEARCH_ENABLED = True  # Enable vector-based semantic search
SEMANTIC_SEARCH_LIMIT = 100  # Number of candidates from vector search (increased for diversity)
MIN_SEMANTIC_SCORE = 0.1  # Minimum similarity score (lowered for better long-term recall)

# Hybrid Retrieval Configuration (Phase 5+)
HYBRID_RETRIEVAL_ENABLED = True  # Combine semantic + recency retrieval
RECENCY_RETRIEVAL_LIMIT = 50  # Number of recent memories to fetch (bypasses semantic filter)
RECENCY_FALLBACK_SEMANTIC_SCORE = 0.15  # Semantic score assigned to recency-only memories

# Phase 2: Multi-Signal Ranking Weights
# These weights sum to 1.0 for final score calculation
RANKING_WEIGHTS = {
    "semantic": 0.5,   # Semantic similarity score weight
    "type": 0.25,      # Memory type priority weight
    "recency": 0.25,   # Recency score weight
}

# Memory type priorities for ranking (higher = more important)
TYPE_PRIORITIES = {
    "constraint": 1.0,    # Constraints are critical
    "instruction": 0.95,  # Instructions are very important
    "preference": 0.7,    # Preferences are valuable
    "entity": 0.6,        # Entities for context
    "commitment": 0.8,    # Commitments are time-sensitive
    "fact": 0.5,          # Facts are general info
    "event": 0.4,         # Events are context
}

# Recency decay configuration
# FIXED: Was 0.1 (too aggressive - 1000 turns = 0 score)
# Now 0.001 (gentler - 1000 turns = 0.37 score)
RECENCY_DECAY_RATE = 0.001  # Decay factor per turn (exponential decay)
RECENCY_MAX_TURNS = 5000    # After this many turns, recency score approaches 0

# Redis Key Prefixes
REDIS_MEMORY_PREFIX = "mem:"
REDIS_DEDUP_PREFIX = "dedup:"
REDIS_TYPE_INDEX_PREFIX = "type:"
REDIS_RECENCY_INDEX = "recent_memories"

# ---------------------------------------------------------------------------
# ARCHITECTURE PROFILE
# ---------------------------------------------------------------------------
# Memora was designed for a single user talking about themselves in the first person
# ("I prefer X", "my name is Y"). A large class of real workloads -- and every
# conversational-memory benchmark -- is instead MULTI-PARTY and TIME-ANCHORED: several
# speakers recounting events that happened on particular dates, across many sessions.
#
# Three assumptions break in that setting, all of them silently:
#
#   1. No speaker.     process_turn() takes only text, so facts about different people
#                      merge into one undifferentiated pool. "Who did X" becomes
#                      unanswerable and attribution is quietly wrong.
#   2. No event time.  `timestamp` is datetime.now() at ingest, not when the thing
#                      happened, so "when" questions cannot be answered at all.
#   3. Global key dedup. `dedup:<type>:<key>` is keyed on (type, key) with no speaker and
#                      no value, so the FIRST `entity:location` permanently blocks every
#                      later one -- a second speaker's location, or the same speaker
#                      moving house, is discarded rather than stored.
#
# MEMORA_PROFILE selects between:
#   "conversation" (default) -- speaker- and time-aware; the architecture below.
#   "legacy"                 -- exactly the pre-redesign behaviour, for A/B comparison.
#
# Everything the profile switches on is individually overridable, so any published number
# can state precisely which mechanisms were active.
MEMORA_PROFILE = os.getenv("MEMORA_PROFILE", "conversation").strip().lower()
_CONV = MEMORA_PROFILE == "conversation"


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Memory Record Fields
MEMORY_FIELDS = [
    "memory_id",
    "type",
    "key",
    "value",
    "confidence",
    "turn_number",
    "timestamp",
    "source_text",
    # Conversation-model fields. `speaker` is who the memory is ABOUT (the utterer);
    # `event_date` is the human-readable date the content refers to, and `event_ts` its
    # epoch form for range queries. All three are distinct from `timestamp`, which
    # remains ingest wall-clock.
    "speaker",
    "event_date",
    "event_ts",
    "mention_count",     # Phase 3: Track repetitions for confidence boost
    "superseded_by",     # Phase 3: ID of memory that supersedes this one
    "supersedes",        # Phase 3: ID of memory this one supersedes
    "is_update",         # Phase 3: Flag if this is an update to existing memory
    "last_accessed_turn",  # Phase 3: For frequency tracking
    # Phase 4 fields
    "access_count",      # Phase 4: Total number of times retrieved
    "merged_from",       # Phase 4: IDs of memories merged into this one
    "promoted_to_core",  # Phase 4: Flag if promoted to core memory
    "decay_applied",     # Phase 4: Total decay applied to confidence
]

# Phase 3: Stage 3 LLM Extraction Configuration
STAGE_3_ENABLED = True  # Enable/disable LLM-based extraction
LLM_PROVIDER = "groq"  # "openai" | "anthropic" | "groq"
LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Support multiple Groq API keys for rate limit rotation.
#
# Groq enforces rate limits at the ORGANIZATION level, not per key, so extra keys from the
# same account share one quota and buy nothing. Keys from separate accounts each carry
# their own quota -- that is the only way more keys increase throughput.
#
# Scanned dynamically (GROQ_API_KEY, then GROQ_API_KEY_1..N until a gap) instead of the
# previous hardcoded 4 slots, which silently ignored any key past GROQ_API_KEY_3.
def _collect_groq_keys(max_keys: int = 64) -> list:
    keys, seen = [], set()
    candidates = [os.getenv("GROQ_API_KEY")]
    for i in range(1, max_keys + 1):
        candidates.append(os.getenv(f"GROQ_API_KEY_{i}"))
    for key in candidates:
        if not key:
            continue
        key = key.strip()
        # Skip the placeholder shipped in .env.example so a half-filled .env fails loudly
        # at "no key configured" rather than obscurely at "401 invalid api key".
        if not key or key.startswith("gsk-your") or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


GROQ_API_KEYS = _collect_groq_keys()
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else None  # Backward compatibility

STAGE_3_CONFIDENCE_THRESHOLD = 0.7  # Escalate to LLM if Stage 2 < this
STAGE_3_MAX_TOKENS = 500  # Max tokens for LLM extraction response (increased to prevent JSON cutoffs)
STAGE_3_TEMPERATURE = 0.1  # Low temperature for consistent extraction

# Stage 3 retry policy (rate limits / transient upstream errors).
# Attempts are floored at STAGE_3_MAX_ATTEMPTS regardless of how many API keys are
# configured. Before this existed, the attempt count was len(api_keys), so a single-key
# deployment made ONE attempt with no backoff and any transient 429 propagated -- fine for
# a demo, fatal for a batch run making tens of thousands of calls.
STAGE_3_MAX_ATTEMPTS = int(os.getenv("STAGE_3_MAX_ATTEMPTS", "6"))
STAGE_3_BACKOFF_BASE = float(os.getenv("STAGE_3_BACKOFF_BASE", "1.0"))  # seconds
STAGE_3_BACKOFF_MAX = float(os.getenv("STAGE_3_BACKOFF_MAX", "60.0"))  # seconds

# Phase 3: Semantic Deduplication Configuration
SEMANTIC_DEDUP_ENABLED = True  # Enable/disable semantic deduplication
SEMANTIC_DEDUP_THRESHOLD = 0.92  # Similarity score to consider duplicate
SEMANTIC_DEDUP_CHECK_LIMIT = 5  # Check top N similar memories for duplicates

# Phase 3: Confidence Scoring Configuration
MIN_CONFIDENCE_TO_STORE = 0.6  # Discard memories below this confidence
HIGH_CONFIDENCE_THRESHOLD = 0.9  # Candidate for core memory promotion
CONFIDENCE_BOOST_PER_MENTION = 0.1  # Boost confidence when repeated
MAX_CONFIDENCE = 0.95  # Maximum confidence after boosts
LOW_CONFIDENCE_DECAY_RATE = 0.1  # Reduce confidence of unused memories
LOW_CONFIDENCE_DECAY_TURNS = 200  # After this many turns, apply decay

# Phase 3: Update Detection Patterns
UPDATE_PATTERNS = [
    r"actually[,\s]+(.+)",
    r"i changed my mind[,\s]+(.+)",
    r"not anymore[,\s]+(.+)",
    r"i used to .+ but now (.+)",
    r"correction[,:\s]+(.+)",
    r"i meant[,:\s]+(.+)",
    r"let me correct that[,:\s]+(.+)",
]

# Phase 3: Confidence Modifiers
CONFIDENCE_MODIFIERS = {
    # Certainty boosters
    "always": 0.1,
    "never": 0.1,
    "definitely": 0.1,
    "absolutely": 0.1,
    "must": 0.1,
    
    # Certainty reducers
    "maybe": -0.2,
    "perhaps": -0.2,
    "possibly": -0.2,
    "might": -0.2,
    "sometimes": -0.15,
    "occasionally": -0.15,
    "could": -0.15,
}

# ============================================================================
# Phase 4: Consolidation & 5-Signal Ranking Configuration
# ============================================================================

# Background Consolidation Worker
CONSOLIDATION_ENABLED = True
CONSOLIDATION_INTERVAL_TURNS = 50  # Run consolidation every N turns
CONSOLIDATION_MIN_MEMORIES = 10    # Minimum memories before consolidation runs

# Memory Decay Configuration
MEMORY_DECAY_ENABLED = False       # DISABLED to test promotion without decay
DECAY_TURNS_THRESHOLD = 500        # Apply decay after memory is this old (was 300)
DECAY_INACTIVE_TURNS = 300         # Extra decay if not accessed in N turns (was 150)
DECAY_RATE_PER_100_TURNS = 0.005   # Confidence reduction per 100 turns (was 0.02 - now 20x slower than original!)
MIN_DECAY_CONFIDENCE = 0.05        # Memories below this may be deleted (was 0.1)
DELETE_VERY_LOW_CONFIDENCE = True  # Auto-delete memories below min threshold

# Memory Merging Configuration
MEMORY_MERGE_ENABLED = True
MERGE_SIMILARITY_THRESHOLD = 0.85  # Similarity above this triggers merge check
MERGE_SAME_TYPE_ONLY = True        # Only merge memories of same type
MAX_MERGED_VALUE_LENGTH = 500      # Max chars for merged value

# Core Memory Promotion Configuration
PROMOTION_ENABLED = True
PROMOTION_CONFIDENCE_THRESHOLD = 0.90  # Min confidence for promotion
PROMOTION_MENTION_THRESHOLD = 3        # Min mentions for promotion
PROMOTION_ACCESS_THRESHOLD = 5         # Min access count for promotion
PROMOTION_AGE_THRESHOLD = 50           # Min turns old for promotion
PROMOTABLE_TYPES = ["entity", "preference", "constraint", "instruction"]

# 5-Signal Ranking Weights (must sum to 1.0)
# Rebalanced to reduce semantic dominance and prioritize memory type diversity
# Frequency reduced to minimal - it creates circular dependency (accessed memories rank higher)
#
# ENV-OVERRIDABLE so the benchmark can ablate the weighting without editing this file.
# Defaults are unchanged, so the baseline measurement is exactly the shipped behaviour.
#
# Why this is worth ablating: `type` (0.40) currently outweighs `semantic` (0.30). That is
# defensible for an assistant, where a dietary constraint matters whether or not the user
# just mentioned it. It is questionable for question answering, where the answer to
# "when did Melanie visit the museum" lives in a `fact` (0.5) or `event` (0.4) -- the two
# LOWEST type priorities -- while an irrelevant `constraint` (1.0) collects 0.40 from type
# alone and can outrank a well-matching event. See BENCHMARK_FINDINGS.md.
def _rank_weight(name: str, default: float) -> float:
    return float(os.getenv(f"RANK_W_{name.upper()}", str(default)))


#
# The conversation profile inverts the type/semantic balance. Type priority answers
# "how important is this KIND of memory in general", which is the right question when
# deciding what an assistant must never forget, and the wrong one when deciding which
# memory answers the question in front of you. Under the legacy weights an entirely
# irrelevant `constraint` scores 0.40 from type alone and outranks a well-matching
# `event` at 0.34 -- and conversational answers live almost entirely in `fact` and
# `event`, the two lowest priorities.
_W = {
    "conversation": {"semantic": 0.55, "type": 0.10, "recency": 0.10,
                     "frequency": 0.05, "confidence": 0.20},
    "legacy": {"semantic": 0.30, "type": 0.40, "recency": 0.10,
               "frequency": 0.05, "confidence": 0.15},
}[("conversation" if _CONV else "legacy")]

RANKING_WEIGHTS_5_SIGNAL = {
    "semantic": _rank_weight("semantic", _W["semantic"]),
    "type": _rank_weight("type", _W["type"]),
    "recency": _rank_weight("recency", _W["recency"]),
    "frequency": _rank_weight("frequency", _W["frequency"]),
    "confidence": _rank_weight("confidence", _W["confidence"]),
}

# Render the originating date alongside each retrieved memory.
#
# Memories are formatted as "- key: value [turn N, X% confident]". A turn index is
# meaningless to a reader answering "when did this happen", and the stored `timestamp` is
# datetime.now() at INGEST time, not the time the conversation describes. So temporal
# questions are unanswerable from the context regardless of how good retrieval is.
#
# When enabled, the date is recovered from the leading "[...]" of `source_text`, which the
# benchmark adapter puts there. Off by default: it changes the prompt for every caller, and
# on a live assistant the ingest date and the event date usually coincide anyway.
MEMORY_CONTEXT_INCLUDE_DATE = _flag("MEMORY_CONTEXT_INCLUDE_DATE", _CONV)

# ---------------------------------------------------------------------------
# CONVERSATION-PROFILE MECHANISMS
# ---------------------------------------------------------------------------

# Dedup identity. Legacy: (type, key) -- globally unique forever, so distinct facts
# sharing a key annihilate each other. Conversation: (type, key, speaker, value), so a
# repeat of the SAME statement still dedups, while a different speaker or a changed value
# is preserved as its own memory. Superseding (Phase 3) remains the mechanism for "this
# replaces that"; exact-key dedup should never have been doing that job implicitly.
DEDUP_KEY_INCLUDES_SPEAKER = _flag("DEDUP_KEY_INCLUDES_SPEAKER", _CONV)
DEDUP_KEY_INCLUDES_VALUE = _flag("DEDUP_KEY_INCLUDES_VALUE", _CONV)

# Semantic dedup must not merge the same kind of event across different dates -- two
# museum visits months apart are two memories, and collapsing them destroys exactly the
# distinctions a temporal or multi-hop question probes.
SEMANTIC_DEDUP_RESPECTS_DATE = _flag("SEMANTIC_DEDUP_RESPECTS_DATE", _CONV)

# Embedding text. Legacy embeds "key | value | type: event" -- a terse fragment with a
# constant noise suffix, compared against natural-language questions. Conversation embeds
# a natural sentence including speaker and date, which sits far closer to query phrasing
# in the same vector space.
EMBED_NATURAL_TEXT = _flag("EMBED_NATURAL_TEXT", _CONV)

# Lexical (BM25) retrieval fused with dense search by Reciprocal Rank Fusion.
# Dense embeddings from a 384-dim MiniLM are weak on rare proper nouns -- exactly the
# names, places and titles that conversational questions turn on. BM25 is excellent at
# them. The two fail differently, which is what makes fusing them worth more than tuning
# either alone.
LEXICAL_SEARCH_ENABLED = _flag("LEXICAL_SEARCH_ENABLED", _CONV)
LEXICAL_SEARCH_LIMIT = int(os.getenv("LEXICAL_SEARCH_LIMIT", "100"))
RRF_K = int(os.getenv("RRF_K", "60"))  # standard RRF damping constant
FUSION_WEIGHT_DENSE = float(os.getenv("FUSION_WEIGHT_DENSE", "1.0"))
FUSION_WEIGHT_LEXICAL = float(os.getenv("FUSION_WEIGHT_LEXICAL", "1.0"))

# Context rendering. Legacy groups memories by TYPE, which destroys chronology -- the
# reader sees preferences, then entities, then events, with no way to order them. For
# reasoning over "what happened when", and for chaining facts across sessions, temporal
# order carries most of the signal.
CONTEXT_CHRONOLOGICAL = _flag("CONTEXT_CHRONOLOGICAL", _CONV)
CONTEXT_INCLUDE_SPEAKER = _flag("CONTEXT_INCLUDE_SPEAKER", _CONV)

# Include the originating utterance for the highest-ranked memories. key:value is a lossy
# compression of what was said; for the top few memories the raw sentence often carries
# the detail the question actually asks for. Bounded so it cannot dominate the budget.
CONTEXT_EVIDENCE_TOP_N = int(os.getenv("CONTEXT_EVIDENCE_TOP_N", "8" if _CONV else "0"))
CONTEXT_EVIDENCE_MAX_CHARS = int(os.getenv("CONTEXT_EVIDENCE_MAX_CHARS", "220"))

# Query-aware retrieval. When a query asks WHEN something happened, memories carrying a
# date are more useful than those without; when it names a person, memories attributed to
# that person are. Both are properties of the QUERY, computed generically -- no dataset
# vocabulary, no per-benchmark rules.
QUERY_AWARE_RETRIEVAL = _flag("QUERY_AWARE_RETRIEVAL", _CONV)
TEMPORAL_INTENT_BOOST = float(os.getenv("TEMPORAL_INTENT_BOOST", "0.15"))
SPEAKER_MATCH_BOOST = float(os.getenv("SPEAKER_MATCH_BOOST", "0.15"))

# Second retrieval pass seeded with entities found in the first. Multi-hop questions name
# one entity and ask about another reachable only through it; a single similarity lookup
# against the original question cannot cross that gap.
MULTIHOP_EXPANSION_ENABLED = _flag("MULTIHOP_EXPANSION_ENABLED", _CONV)
MULTIHOP_SEED_MEMORIES = int(os.getenv("MULTIHOP_SEED_MEMORIES", "5"))
MULTIHOP_EXTRA_LIMIT = int(os.getenv("MULTIHOP_EXTRA_LIMIT", "30"))

# Frequency scoring configuration
FREQUENCY_DECAY_RATE = 0.05         # How fast frequency score decays
FREQUENCY_MAX_ACCESSES = 20         # Normalize access count against this
ACCESS_RECENCY_WEIGHT = 0.6         # Weight for recent accesses vs total count

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
