"""
Memory Retrieval - Phase 1, 2, 3 & 4
Type-based + recency-based retrieval with semantic search (Phase 2), 
superseding filter (Phase 3), and 5-signal ranking (Phase 4)
"""

import logging
import math
from typing import List, Dict, Optional

from .config import (
    MAX_MEMORIES_TO_RETRIEVE,
    MEMORY_TOKEN_BUDGET,
    MEMORY_TYPES,
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
        
        if self._semantic_enabled:
            ranking_mode = "5-signal" if use_5_signal else "3-signal"
            logger.info(f"Semantic search enabled for memory retrieval ({ranking_mode} ranking)")
        else:
            logger.info("Using non-semantic retrieval (Phase 1 mode)")
    
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
        always_on_types = ["constraint", "instruction"]
        for mem_type in always_on_types:
            memories = self.redis_store.get_memories_by_type(mem_type, limit=20)
            for mem in memories:
                mem_id = mem['memory_id']
                if mem_id not in all_memories:
                    mem['semantic_score'] = 0.5  # Default score for always-on
                    all_memories[mem_id] = mem
                # Boost semantic score for always-on types already found
                elif mem_id in all_memories:
                    all_memories[mem_id]['semantic_score'] = max(
                        all_memories[mem_id].get('semantic_score', 0), 
                        0.5
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
        
        sections = {
            "constraint": [],
            "instruction": [],
            "preference": [],
            "entity": [],
            "commitment": [],
            "fact": [],
            "event": [],
        }
        
        # Group by type
        for mem in memories:
            mem_type = mem.get('type', 'fact')
            if mem_type in sections:
                sections[mem_type].append(mem)
        
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
                line = f"- {key}: {value} [turn {turn}, {confidence*100:.0f}% confident]"
                section_lines.append(line)
            
            formatted_sections.append("\n".join(section_lines))
        
        return "\n\n".join(formatted_sections)

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
