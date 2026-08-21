"""
Memory Retrieval - Phase 1, 2, 3 & 4
Type-based + recency-based retrieval with semantic search (Phase 2), 
superseding filter (Phase 3), and 5-signal ranking (Phase 4)
"""

import logging
import math
import re
from typing import List, Dict, Optional

from .lexical_index import (
    BM25Index,
    normalize_scores,
    reciprocal_rank_fusion,
)
from .config import (
    ALWAYS_ON_SEMANTIC_FLOOR,
    BM25_B,
    BM25_K1,
    CONTEXT_CHRONOLOGICAL,
    DATE_MATCH_BOOST,
    CONTEXT_EVIDENCE_MAX_CHARS,
    CONTEXT_EVIDENCE_TOP_N,
    CONTEXT_INCLUDE_SPEAKER,
    ENTITY_INDEX_ENABLED,
    ENTITY_MATCH_SCORE,
    ENTITY_RETRIEVAL_LIMIT,
    FUSION_WEIGHT_DENSE,
    FUSION_WEIGHT_LEXICAL,
    LEXICAL_SEARCH_ENABLED,
    LEXICAL_SEARCH_LIMIT,
    MAX_MEMORIES_TO_RETRIEVE,
    MEMORY_CONTEXT_INCLUDE_DATE,
    MEMORY_TOKEN_BUDGET,
    MEMORY_TYPES,
    MULTIHOP_EXPANSION_ENABLED,
    MULTIHOP_EXTRA_LIMIT,
    MULTIHOP_SEED_MEMORIES,
    QUERY_AWARE_RETRIEVAL,
    RRF_K,
    SPEAKER_MATCH_BOOST,
    TEMPORAL_INTENT_BOOST,
    SEMANTIC_SEARCH_ENABLED,
    SEMANTIC_SEARCH_LIMIT,
    MIN_SEMANTIC_SCORE,
    RANKING_WEIGHTS,
    TYPE_PRIORITIES,
    RECENCY_DECAY_RATE,
    RECENCY_MAX_TURNS,
    # Phase 4 imports
    RANKING_WEIGHTS_5_SIGNAL,
    FREQUENCY_DECAY_RATE,
    FREQUENCY_MAX_ACCESSES,
    ACCESS_RECENCY_WEIGHT,
    # Hybrid retrieval imports
    HYBRID_RETRIEVAL_ENABLED,
    RECENCY_RETRIEVAL_LIMIT,
    RECENCY_FALLBACK_SEMANTIC_SCORE,
)
from .redis_store import RedisStore

logger = logging.getLogger(__name__)

# Leading "[8 May, 2023]"-style stamp that the LoCoMo adapter prefixes onto each turn
# before handing it to process_turn(). Matched against `source_text`, which store_memory
# already persists, so dates can be surfaced without re-ingesting anything.
_SOURCE_DATE_RE = re.compile(r"^\s*\[([^\]]{3,40})\]")


# Query-intent detection. Deliberately about the SHAPE of the question rather than any
# dataset's vocabulary: "when/what year/how long ago" marks a query whose answer is a time,
# for which a memory carrying a date is more useful than one without. No benchmark-specific
# terms appear here and none should.
_TEMPORAL_INTENT_RE = re.compile(
    r"\b(when|what\s+(?:date|time|year|month|day)|how\s+long|how\s+many\s+"
    r"(?:days|weeks|months|years)|before|after|earlier|later|first|last|"
    r"recently|ago|since|until|during)\b",
    re.IGNORECASE,
)

# Capitalised tokens that are not sentence-initial: a cheap, language-agnostic proxy for
# named entities that needs no NER model in the hot path.
_CAPITALISED_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]{2,})\b")


# Dates a query names. Generic calendar vocabulary -- years and month names -- not any
# dataset's formatting. Language-specific (English), like the stopword list and the
# temporal-intent patterns above; not benchmark-specific.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_MONTH_ABBR = tuple(m[:3] for m in _MONTHS)


def has_temporal_intent(query: str) -> bool:
    return bool(_TEMPORAL_INTENT_RE.search(query or ""))


def query_dates(query: str) -> set:
    """Year and month tokens named in the query, lowercased.

    A question asking "when" is served by any dated memory; a question naming *May 2023*
    is served by memories from May 2023 specifically. Without this, temporal scoring
    cannot tell those apart -- every dated memory gets the same credit.
    """
    text = (query or "").lower()
    found = {m.group(0) for m in _YEAR_RE.finditer(text)}
    for full, abbr in zip(_MONTHS, _MONTH_ABBR):
        if re.search(rf"\b{abbr}[a-z]*\b", text) and (full in text or abbr in text):
            found.add(abbr)
    return found


def date_overlap(query_tokens: set, event_date: str) -> bool:
    """True when a memory's date shares a year or month with the query's."""
    if not query_tokens or not event_date:
        return False
    d = event_date.lower()
    for tok in query_tokens:
        if tok.isdigit():
            if tok in d:
                return True
        elif re.search(rf"\b{tok}[a-z]*\b", d):
            return True
    return False


def query_entities(query: str) -> set:
    """Probable named entities in a query, lowercased."""
    return {m.group(1).lower() for m in _CAPITALISED_RE.finditer(query or "")}


# "1:56 pm on 8 May, 2023" -> "8 May, 2023". Time-of-day is never what a conversational
# question asks about, and repeating it on every one of ~50 context lines crowds out the
# part that matters. Dropped from the rendered context only; the stored field is untouched.
_TIME_PREFIX_RE = re.compile(
    r"^\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+on\s+", re.IGNORECASE)


def _clean_date(raw: str) -> str:
    return _TIME_PREFIX_RE.sub("", (raw or "").strip()).strip()


def _strip_redundant_speaker(key: str, speaker: str) -> str:
    """Drop a leading speaker name from a memory key.

    Speaker-aware extraction is told to name the subject, so Stage 3 returns keys like
    "Caroline current activity". Rendering already prefixes the speaker, giving
    "Caroline - Caroline current activity: ..." on every line. The duplication wastes
    context budget and reads as though two different Carolines are involved.
    """
    if not key or not speaker:
        return key
    first = speaker.split()[0].lower()
    for candidate in (speaker.lower(), first):
        if candidate and key.lower().startswith(candidate + " "):
            return key[len(candidate) + 1:].lstrip("-:' ").strip() or key
        if candidate and key.lower().startswith(candidate + "'s "):
            return key[len(candidate) + 3:].lstrip("-:' ").strip() or key
    return key


def _source_date(memory: Dict) -> Optional[str]:
    """Date this memory came from, or None.

    process_turn() accepts no timestamp argument, so callers that care about when
    something happened fold it into the message text. `timestamp` on the memory is
    datetime.now() at ingest, which for replayed or backfilled conversations is not the
    date the content is about -- hence reading it back out of the source text instead.
    """
    src = memory.get('source_text') or ''
    match = _SOURCE_DATE_RE.match(src)
    return match.group(1).strip() if match else None


class MemoryRetriever:
    """
    Retrieves relevant memories for prompt injection.
    
    Phase 1: Type priority + recency-based retrieval
    Phase 2: Adds semantic search + multi-signal ranking (3 signals)
    Phase 3: Filters superseded memories
    Phase 4: 5-signal ranking (semantic + type + recency + frequency + confidence)
    """

    def __init__(self, redis_store: RedisStore, vector_store=None, use_5_signal: bool = True, user_id: Optional[str] = None):
        """
        Initialize the retriever.
        
        Args:
            redis_store: Redis store instance
            vector_store: Optional vector store for semantic search (Phase 2)
            use_5_signal: Use 5-signal ranking (Phase 4) instead of 3-signal
            user_id: User ID for multi-user isolation
        """
        self.redis_store = redis_store
        self.vector_store = vector_store
        self.user_id = user_id  # Store user_id for filtering
        self._semantic_enabled = SEMANTIC_SEARCH_ENABLED and vector_store is not None
        self._use_5_signal = use_5_signal

        # Lexical index, rebuilt when the store size changes. BM25 corpus statistics
        # (average document length, document frequencies) shift with every write, and for
        # a few thousand memories a rebuild is milliseconds -- cheaper than maintaining
        # incremental correctness.
        self._bm25 = None
        self._bm25_size = -1

        if self._semantic_enabled:
            ranking_mode = "5-signal" if use_5_signal else "3-signal"
            logger.info(f"Semantic search enabled for memory retrieval ({ranking_mode} ranking)")
        else:
            logger.info("Using non-semantic retrieval (Phase 1 mode)")
    
    def _lexical_search(self, query: str, limit: int) -> List:
        """BM25 hits for `query`, rebuilding the index if the store has changed.

        Any failure here degrades to "no lexical channel" rather than breaking retrieval:
        fusion is an improvement over dense-only search, never a dependency of it.
        """
        try:
            size = self.redis_store.count_memories()
            if self._bm25 is None or size != self._bm25_size:
                memories = self.redis_store.get_all_memories()
                if self.user_id:
                    memories = [m for m in memories
                                if not m.get('user_id') or m.get('user_id') == self.user_id]
                self._bm25 = BM25Index(k1=BM25_K1, b=BM25_B).build(memories)
                self._bm25_size = size
                logger.debug("Rebuilt lexical index over %d memories", len(self._bm25))
            return self._bm25.search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lexical search unavailable, continuing dense-only: %s", exc)
            return []

    def _expand_multihop(self, query: str, all_memories: Dict) -> None:
        """Second retrieval pass seeded with the best first-pass memories.

        Mutates `all_memories` in place, adding candidates at a damped score so a bridged
        hit can compete without displacing directly-relevant memories.
        """
        try:
            seeds = sorted(
                all_memories.values(),
                key=lambda m: float(m.get('semantic_score', 0) or 0),
                reverse=True,
            )[:MULTIHOP_SEED_MEMORIES]
            if not seeds:
                return

            seed_text = " ".join(
                f"{m.get('key', '')} {m.get('value', '')}" for m in seeds
            ).strip()
            if not seed_text:
                return

            expanded = f"{query} {seed_text}"
            hits = self.vector_store.search_similar(
                query=expanded,
                limit=MULTIHOP_EXTRA_LIMIT,
                min_score=MIN_SEMANTIC_SCORE,
                user_id=self.user_id,
            )

            added = 0
            for result in hits:
                mem_id = result['memory_id']
                if mem_id in all_memories:
                    continue
                full = self.redis_store.get_memory(mem_id) or result['memory']
                # Damped: reached indirectly, so it should rank below anything the
                # original question matched directly.
                full['semantic_score'] = float(result['score']) * 0.6
                full['multihop_hop'] = 2
                all_memories[mem_id] = full
                added += 1
            if added:
                logger.debug("Multi-hop expansion added %d candidates", added)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multi-hop expansion skipped: %s", exc)

    def _filter_superseded(self, memories: List[Dict]) -> List[Dict]:
        """
        Filter out memories that have been superseded.
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            Filtered list without superseded memories
        """
        active = []
        superseded_count = 0
        
        for mem in memories:
            superseded_by = mem.get('superseded_by')
            # Check if superseded_by exists and is not None or empty string
            if superseded_by and superseded_by not in [None, '', 'None']:
                superseded_count += 1
                logger.debug(f"Filtered superseded memory: {mem['memory_id']}")
            else:
                active.append(mem)
        
        if superseded_count > 0:
            logger.info(f"Filtered out {superseded_count} superseded memories")
        
        return active

    def retrieve(
        self, 
        current_message: str, 
        turn_number: int,
        priority_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant memories for the current turn.
        
        Phase 1: Simple retrieval using type priority + recency
        Phase 2: Semantic similarity search + multi-signal ranking
        
        Args:
            current_message: The current user message
            turn_number: Current turn number
            priority_types: Memory types to prioritize (e.g., ["constraint", "preference"])
        
        Returns:
            List of memory dictionaries, ranked by relevance
        """
        if self._semantic_enabled:
            return self._retrieve_with_semantic_search(
                current_message, turn_number, priority_types
            )
        else:
            return self._retrieve_phase1(
                current_message, turn_number, priority_types
            )
    
    def _retrieve_with_semantic_search(
        self,
        current_message: str,
        turn_number: int,
        priority_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Phase 2+: Hybrid retrieval using semantic search + recency-based retrieval.
        
        Strategy:
        - Semantic branch: Memories with similarity > MIN_SEMANTIC_SCORE
        - Recency branch: Recent memories (bypass semantic filter entirely)
        - Merge both, then apply 5-signal ranking
        
        This ensures old memories with low semantic similarity still get retrieved.
        """
        all_memories = {}  # memory_id -> memory with scores
        
        # Branch 1: Semantic Search (content relevance with filter)
        semantic_results = self.vector_store.search_similar(
            query=current_message,
            limit=SEMANTIC_SEARCH_LIMIT,
            min_score=MIN_SEMANTIC_SCORE,
            user_id=self.user_id,  # Filter by user for multi-user isolation
        )
        
        for result in semantic_results:
            memory = result['memory']
            memory_id = result['memory_id']
            
            # Get full memory from Redis (has more fields)
            full_memory = self.redis_store.get_memory(memory_id)
            if full_memory:
                full_memory['semantic_score'] = result['score']
                all_memories[memory_id] = full_memory
            else:
                # Fall back to vector store payload
                memory['semantic_score'] = result['score']
                all_memories[memory_id] = memory
        
        logger.debug(f"Semantic search found {len(all_memories)} candidates")

        # Branch 1b: LEXICAL (BM25) + Reciprocal Rank Fusion.
        #
        # Dense retrieval and BM25 fail on different queries -- MiniLM is weak on rare
        # proper nouns, which is what BM25 is best at -- so fusing them recalls more than
        # either alone. Fusing by RANK (RRF) rather than by score avoids having to
        # normalise a cosine against a BM25 magnitude, which needs corpus-specific
        # constants that do not transfer.
        dense_ranked = [r['memory_id'] for r in semantic_results]
        if LEXICAL_SEARCH_ENABLED:
            lexical_hits = self._lexical_search(current_message, LEXICAL_SEARCH_LIMIT)
            if lexical_hits:
                lex_ranked = [mid for mid, _ in lexical_hits]
                fused = normalize_scores(reciprocal_rank_fusion(
                    [(dense_ranked, FUSION_WEIGHT_DENSE),
                     (lex_ranked, FUSION_WEIGHT_LEXICAL)],
                    k=RRF_K,
                ))

                # Pull in lexical-only hits: these are the documents dense search missed,
                # and they are the entire reason for running a second channel.
                added = 0
                for mem_id in lex_ranked:
                    if mem_id not in all_memories:
                        full = self.redis_store.get_memory(mem_id)
                        if full:
                            all_memories[mem_id] = full
                            added += 1

                for mem_id, mem in all_memories.items():
                    if mem_id in fused:
                        mem['semantic_score'] = fused[mem_id]
                        mem['fusion_score'] = fused[mem_id]

                logger.debug(
                    "Lexical branch: %d hits, %d new candidates, %d fused",
                    len(lexical_hits), added, len(fused),
                )

        # Branch 1d: ENTITY-CENTRIC RETRIEVAL.
        #
        # When the question names people or places, pull every memory about them. This is
        # the recall path neither dense nor lexical search covers reliably: the memory that
        # answers a multi-hop question often shares almost no wording with the question,
        # so it ranks nowhere on similarity while being obviously relevant by subject.
        #
        # Scored by HOW MANY of the query's entities a memory mentions. A memory naming
        # both entities in the question is the likeliest bridge between them, and that
        # multiplier is where most of the value sits.
        if ENTITY_INDEX_ENABLED and getattr(self.redis_store, "entity_index", None):
            q_ents = query_entities(current_message)
            if q_ents:
                counts = self.redis_store.entity_index.match_counts(q_ents)
                ranked_ents = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                added = 0
                for mem_id, hits in ranked_ents[:ENTITY_RETRIEVAL_LIMIT]:
                    score = min(1.0, ENTITY_MATCH_SCORE * hits)
                    if mem_id in all_memories:
                        # Never lower a score earned by relevance.
                        all_memories[mem_id]['semantic_score'] = max(
                            float(all_memories[mem_id].get('semantic_score', 0) or 0),
                            score)
                        all_memories[mem_id]['entity_hits'] = hits
                    else:
                        full = self.redis_store.get_memory(mem_id)
                        if full:
                            full['semantic_score'] = score
                            full['entity_hits'] = hits
                            all_memories[mem_id] = full
                            added += 1
                if added:
                    logger.debug("Entity branch added %d candidates for %s",
                                 added, sorted(q_ents))

        # Branch 1c: MULTI-HOP EXPANSION.
        #
        # A multi-hop question names one entity and asks about another reachable only
        # through it ("what instrument does the friend she met at the museum play?").
        # One similarity lookup against the original wording cannot bridge that -- the
        # second entity does not appear in the query at all. Seeding a second pass with
        # the content of the best first-pass memories gives the bridge a chance to be
        # found. Generic: no dataset vocabulary, just re-querying with retrieved text.
        if MULTIHOP_EXPANSION_ENABLED and all_memories:
            self._expand_multihop(current_message, all_memories)

        # Branch 2: Recency-Based Retrieval (temporal relevance, NO semantic filter)
        if HYBRID_RETRIEVAL_ENABLED:
            recent_memories = self.redis_store.get_recent_memories(
                limit=RECENCY_RETRIEVAL_LIMIT
            )
            
            recency_added = 0
            for mem in recent_memories:
                mem_id = mem['memory_id']
                if mem_id not in all_memories:
                    # Assign fallback semantic score (neutral, doesn't dominate)
                    mem['semantic_score'] = RECENCY_FALLBACK_SEMANTIC_SCORE
                    all_memories[mem_id] = mem
                    recency_added += 1
                    
            logger.debug(f"Recency branch added {recency_added} new candidates (hybrid mode)")
        
        # Step 2: Always include constraint and instruction types
        #
        # THE FLOOR IS A SECOND PATH FOR THE PROBLEM THE RANKING WEIGHTS FIXED.
        #
        # These memories are injected regardless of relevance, then given a FIXED semantic
        # score. At the legacy floor of 0.5 with the conversation profile's semantic weight
        # of 0.55, an entirely irrelevant constraint collects 0.275 from the relevance
        # signal alone. After RRF normalisation only the top hit scores 1.0 and a
        # single-channel hit around rank 50 normalises to roughly 0.28 -- so rebalancing
        # type 0.40 -> 0.10 achieved little, because the same crowding simply moved into
        # the semantic term.
        #
        # Two changes: the floor is configurable and much lower under the conversation
        # profile, and it no longer OVERWRITES a real score. Raising an already-retrieved
        # memory to the floor flattened genuinely relevant always-on hits onto the same
        # value as irrelevant ones, destroying the ordering among them.
        always_on_types = ["constraint", "instruction"]
        for mem_type in always_on_types:
            memories = self.redis_store.get_memories_by_type(mem_type, limit=20)
            for mem in memories:
                mem_id = mem['memory_id']
                if mem_id not in all_memories:
                    mem['semantic_score'] = ALWAYS_ON_SEMANTIC_FLOOR
                    mem['always_on'] = True
                    all_memories[mem_id] = mem
                else:
                    # Already retrieved on merit: keep the earned score, never lower or
                    # flatten it.
                    all_memories[mem_id]['semantic_score'] = max(
                        float(all_memories[mem_id].get('semantic_score', 0) or 0),
                        ALWAYS_ON_SEMANTIC_FLOOR,
                    )
        
        # Step 3: Add priority types if specified
        if priority_types:
            for mem_type in priority_types:
                if mem_type not in always_on_types:
                    memories = self.redis_store.get_memories_by_type(mem_type, limit=10)
                    for mem in memories:
                        mem_id = mem['memory_id']
                        if mem_id not in all_memories:
                            mem['semantic_score'] = 0.3  # Lower default for priority types
                            all_memories[mem_id] = mem
        
        # Step 4: Calculate multi-signal ranking scores
        # Computed once per query, not per memory.
        temporal_intent = QUERY_AWARE_RETRIEVAL and has_temporal_intent(current_message)
        q_entities = query_entities(current_message) if QUERY_AWARE_RETRIEVAL else set()
        q_dates = query_dates(current_message) if QUERY_AWARE_RETRIEVAL else set()

        ranked_memories = []
        
        for memory_id, memory in all_memories.items():
            # Semantic score (0-1)
            semantic_score = memory.get('semantic_score', 0)
            
            # Type priority score (0-1)
            mem_type = memory.get('type', 'fact')
            type_score = TYPE_PRIORITIES.get(mem_type, 0.5)
            
            # Recency score (0-1, exponential decay)
            mem_turn = int(memory.get('turn_number', 0))
            turns_ago = max(0, turn_number - mem_turn)
            recency_score = math.exp(-RECENCY_DECAY_RATE * turns_ago)
            
            if self._use_5_signal:
                # Phase 4: 5-signal ranking
                # Frequency score (0-1) - based on access count and recency
                access_count = int(memory.get('access_count', 0))
                last_accessed = int(memory.get('last_accessed_turn', mem_turn))
                access_recency = max(0, turn_number - last_accessed)
                
                # Normalize access count
                normalized_access = min(1.0, access_count / FREQUENCY_MAX_ACCESSES)
                # Decay based on recency of last access
                access_recency_factor = math.exp(-FREQUENCY_DECAY_RATE * access_recency)
                # Combine count and recency
                frequency_score = (
                    (1 - ACCESS_RECENCY_WEIGHT) * normalized_access +
                    ACCESS_RECENCY_WEIGHT * access_recency_factor * normalized_access
                )
                
                # Confidence score (0-1) - from memory's stored confidence
                confidence_score = float(memory.get('confidence', 0.5))
                
                # Combined score using 5-signal weighted sum
                final_score = (
                    RANKING_WEIGHTS_5_SIGNAL['semantic'] * semantic_score +
                    RANKING_WEIGHTS_5_SIGNAL['type'] * type_score +
                    RANKING_WEIGHTS_5_SIGNAL['recency'] * recency_score +
                    RANKING_WEIGHTS_5_SIGNAL['frequency'] * frequency_score +
                    RANKING_WEIGHTS_5_SIGNAL['confidence'] * confidence_score
                )
                
                memory['frequency_score'] = frequency_score
                memory['confidence_score'] = confidence_score
            else:
                # Phase 2/3: 3-signal ranking
                final_score = (
                    RANKING_WEIGHTS['semantic'] * semantic_score +
                    RANKING_WEIGHTS['type'] * type_score +
                    RANKING_WEIGHTS['recency'] * recency_score
                )
            
            # QUERY-AWARE ADJUSTMENT.
            #
            # Both boosts are properties of the QUERY, computed generically:
            #   * a question asking WHEN is better served by a memory that carries a date;
            #   * a question naming a person is better served by that person's memories.
            # Additive and small, so they reorder near-ties rather than overriding
            # relevance. No dataset vocabulary is involved.
            if QUERY_AWARE_RETRIEVAL:
                mem_date = (memory.get('event_date') or '').strip()
                if temporal_intent and mem_date:
                    final_score += TEMPORAL_INTENT_BOOST
                # Carrying the RIGHT date beats carrying any date. Stacks with the intent
                # boost, because a dated memory matching the asked-about period is the
                # strongest temporal evidence available.
                if q_dates and date_overlap(q_dates, mem_date):
                    final_score += DATE_MATCH_BOOST
                if q_entities:
                    speaker = str(memory.get('speaker') or '').strip().lower()
                    if speaker and speaker in q_entities:
                        final_score += SPEAKER_MATCH_BOOST

            memory['retrieval_score'] = final_score
            memory['semantic_score'] = semantic_score
            memory['type_score'] = type_score
            memory['recency_score'] = recency_score

            ranked_memories.append(memory)
        
        # Sort by final score
        ranked_memories.sort(key=lambda m: m['retrieval_score'], reverse=True)
        
        # Filter out superseded memories (Phase 3)
        ranked_memories = self._filter_superseded(ranked_memories)
        
        # Take top K
        top_memories = ranked_memories[:MAX_MEMORIES_TO_RETRIEVE]
        
        # Budget check
        estimated_tokens = len(top_memories) * 50
        if estimated_tokens > MEMORY_TOKEN_BUDGET:
            max_count = MEMORY_TOKEN_BUDGET // 50
            top_memories = top_memories[:max_count]
            logger.warning(f"Trimmed memories to {max_count} to fit token budget")
        
        # Phase 4: Track access counts
        for memory in top_memories:
            self.redis_store.increment_access_count(memory['memory_id'], turn_number)
        
        logger.info(
            f"Retrieved {len(top_memories)} memories (semantic search) for turn {turn_number} "
            f"(from {len(all_memories)} candidates)"
        )
        
        return top_memories
    
    def _retrieve_phase1(
        self,
        current_message: str,
        turn_number: int,
        priority_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Phase 1: Simple retrieval using type priority + recency (fallback).
        """
        all_memories = []
        
        # Strategy 1: Always retrieve CONSTRAINT and INSTRUCTION types
        # These are critical and should always be considered
        always_on_types = ["constraint", "instruction"]
        
        for mem_type in always_on_types:
            memories = self.redis_store.get_memories_by_type(mem_type, limit=20)
            for mem in memories:
                mem['retrieval_score'] = 1.0  # Max priority for always-on
            all_memories.extend(memories)
        
        # Strategy 2: Get recent memories (recency-based)
        recent_memories = self.redis_store.get_recent_memories(limit=30)
        
        # Score recent memories by recency
        for i, mem in enumerate(recent_memories):
            # Recency score: exponential decay
            # Most recent = 1.0, decays as we go back
            recency_score = 0.9 ** i
            
            # If already in always_on, don't add again
            if mem['type'] not in always_on_types:
                mem['retrieval_score'] = recency_score * 0.5  # Lower than always-on
                all_memories.append(mem)
        
        # Strategy 3: Priority types (if specified)
        if priority_types:
            for mem_type in priority_types:
                if mem_type not in always_on_types:
                    memories = self.redis_store.get_memories_by_type(mem_type, limit=10)
                    for mem in memories:
                        # Check if already retrieved
                        if not any(m['memory_id'] == mem['memory_id'] for m in all_memories):
                            mem['retrieval_score'] = 0.7  # Medium priority
                            all_memories.append(mem)
        
        # Deduplicate by memory_id (keep highest score)
        seen = {}
        for mem in all_memories:
            mem_id = mem['memory_id']
            if mem_id not in seen or mem['retrieval_score'] > seen[mem_id]['retrieval_score']:
                seen[mem_id] = mem
        
        all_memories = list(seen.values())
        
        # Rank by retrieval_score
        all_memories.sort(key=lambda m: m['retrieval_score'], reverse=True)
        
        # Filter out superseded memories (Phase 3)
        all_memories = self._filter_superseded(all_memories)
        
        # Take top K
        top_memories = all_memories[:MAX_MEMORIES_TO_RETRIEVE]
        
        # Budget check (estimate ~50 tokens per memory on average)
        # This is a rough estimate; Phase 2+ will have more precise token counting
        estimated_tokens = len(top_memories) * 50
        
        if estimated_tokens > MEMORY_TOKEN_BUDGET:
            # Trim to fit budget
            max_count = MEMORY_TOKEN_BUDGET // 50
            top_memories = top_memories[:max_count]
            logger.warning(f"Trimmed memories to {max_count} to fit token budget")
        
        logger.info(
            f"Retrieved {len(top_memories)} memories for turn {turn_number} "
            f"(from {len(all_memories)} candidates)"
        )
        
        return top_memories

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        Format retrieved memories for injection into the prompt.
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            Formatted string ready for prompt injection
        """
        if not memories:
            return ""

        if CONTEXT_CHRONOLOGICAL:
            return self._format_chronological(memories)

        sections = {
            "constraint": [],
            "instruction": [],
            "preference": [],
            "entity": [],
            "commitment": [],
            "fact": [],
            "event": [],
        }

        # Group by type.
        #
        # Anything with an unrecognised type is bucketed under "fact" rather than dropped.
        # It used to be skipped silently, which meant a memory could be retrieved, counted
        # in `retrieved_count`, have its access_count incremented -- and never reach the
        # prompt. Stage 3 is asked for one of MEMORY_TYPES but an LLM can return something
        # else, and the resulting loss was invisible in every statistic.
        for mem in memories:
            mem_type = mem.get('type') or 'fact'
            if mem_type in sections:
                sections[mem_type].append(mem)
            else:
                logger.debug(
                    "Memory %s has unrecognised type %r; formatting it as 'fact'",
                    mem.get('memory_id', '?'), mem_type,
                )
                sections["fact"].append(mem)
        
        # Format each section
        formatted_sections = []
        
        for mem_type, mem_list in sections.items():
            if not mem_list:
                continue
            
            section_title = mem_type.upper()
            section_lines = [f"=== {section_title} ==="]
            
            for mem in mem_list:
                key = mem.get('key', 'unknown')
                value = mem.get('value', '')
                confidence = mem.get('confidence', 0)
                turn = mem.get('turn_number', 0)
                
                # Format: key: value [turn X, confidence Y%]
                # With MEMORY_CONTEXT_INCLUDE_DATE, prefix the date the memory came from
                # so "when" questions are answerable at all.
                stamp = f"turn {turn}"
                if MEMORY_CONTEXT_INCLUDE_DATE:
                    when = _source_date(mem)
                    if when:
                        stamp = f"{when}, turn {turn}"
                line = f"- {key}: {value} [{stamp}, {confidence*100:.0f}% confident]"
                section_lines.append(line)
            
            formatted_sections.append("\n".join(section_lines))
        
        return "\n\n".join(formatted_sections)

    def _format_chronological(self, memories: List[Dict]) -> str:
        """Render memories as a timeline rather than grouped by type.

        Grouping by type puts every preference together, then every entity, then every
        event, and discards ordering entirely. For questions about what happened, when it
        happened, or what followed what, sequence carries most of the signal -- and a
        reader cannot reconstruct it from `turn N` annotations scattered across sections.

        Ordering is by event date where known, falling back to turn order. Memories with
        no date sort last, under their own heading, so undated facts stay available
        without corrupting the timeline.
        """
        dated: List[Dict] = []
        undated: List[Dict] = []
        for mem in memories:
            (dated if (mem.get('event_date') or _source_date(mem)) else undated).append(mem)

        def sort_key(mem: Dict):
            ts = float(mem.get('event_ts') or 0.0)
            return (ts if ts else float('inf'), int(mem.get('turn_number', 0) or 0))

        dated.sort(key=sort_key)
        undated.sort(key=lambda m: int(m.get('turn_number', 0) or 0))

        # Evidence is attached to the highest-RANKED memories, which is why this is
        # computed before the chronological sort reorders them.
        top_ranked = (
            sorted(memories, key=lambda m: float(m.get('retrieval_score', 0) or 0),
                  reverse=True)[:CONTEXT_EVIDENCE_TOP_N]
            if CONTEXT_EVIDENCE_TOP_N > 0 else []
        )
        top_ids = {m.get('memory_id') for m in top_ranked}

        def render(mem: Dict) -> str:
            when = _clean_date(mem.get('event_date') or _source_date(mem) or '')
            who = str(mem.get('speaker') or '').strip()
            key = str(mem.get('key') or '').replace('_', ' ').strip()
            if CONTEXT_INCLUDE_SPEAKER:
                key = _strip_redundant_speaker(key, who)
            value = str(mem.get('value') or '').strip()
            mem_type = str(mem.get('type') or 'fact')

            head = f"[{when}] " if when else ""
            # ASCII separator on purpose: this string lands in prompts, JSON results and
            # log files that get read on consoles with a non-UTF-8 codepage.
            if CONTEXT_INCLUDE_SPEAKER and who:
                head += f"{who} - "
            body = f"{key}: {value}" if key and value else (value or key)
            line = f"- {head}{body} ({mem_type})"

            if mem.get('memory_id') in top_ids:
                src = str(mem.get('source_text') or '').strip()
                if src:
                    # key:value is a lossy compression of what was said; for the few
                    # best-ranked memories the original sentence often holds the exact
                    # detail the question asks for.
                    snippet = " ".join(src.split())
                    if len(snippet) > CONTEXT_EVIDENCE_MAX_CHARS:
                        snippet = snippet[:CONTEXT_EVIDENCE_MAX_CHARS].rstrip() + "..."
                    line += f"\n    said: \"{snippet}\""
            return line

        out: List[str] = []
        if top_ranked:
            # The chronological timeline below can put 50 memories between the reader
            # and the one it needs -- "lost in the middle" is a real attention effect,
            # not a hypothetical one, and it's exactly what surfaced in the smoke test
            # (a correct-but-buried fact losing to a plausible-but-wrong one that
            # happened to sit nearer the top). This section repeats the same top-ranked
            # memories the timeline already marks with a quote, but leads with them in
            # relevance order instead of making the reader find them chronologically.
            out.append("=== MOST RELEVANT (highest ranked, not necessarily latest) ===")
            out.extend(render(m) for m in top_ranked)
            out.append("")
        if dated:
            out.append("=== TIMELINE (oldest first) ===")
            out.extend(render(m) for m in dated)
        if undated:
            out.append("")
            out.append("=== UNDATED ===")
            out.extend(render(m) for m in undated)
        return "\n".join(out)

    def retrieve_for_prompt(
        self,
        current_message: str,
        turn_number: int,
        priority_types: Optional[List[str]] = None,
    ) -> str:
        """
        One-shot method: retrieve and format memories for prompt.
        
        Returns:
            Formatted memory string ready for injection
        """
        memories = self.retrieve(current_message, turn_number, priority_types)
        return self.format_memories_for_prompt(memories)
