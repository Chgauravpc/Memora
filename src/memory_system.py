"""
Main Memory System - Phase 1, 2, 3 & 4
Orchestrates the full pipeline: Extract → Store → Retrieve → Inject
Phase 2 adds: Vector store, embeddings, semantic search, multi-signal ranking
Phase 3 adds: LLM extraction, semantic deduplication, update detection
Phase 4 adds: Background consolidation, memory decay/merge, core promotion, 5-signal ranking
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .flat_file_store import FlatFileStore
from .redis_store import RedisStore
from .extractor import MemoryExtractor
from .retriever import MemoryRetriever
from .config import (
    SEMANTIC_SEARCH_ENABLED,
    SEMANTIC_DEDUP_ENABLED,
    SEMANTIC_DEDUP_THRESHOLD,
    SEMANTIC_DEDUP_CHECK_LIMIT,
    SEMANTIC_DEDUP_RESPECTS_DATE,
    # Phase 4 config
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_INTERVAL_TURNS,
)

logger = logging.getLogger(__name__)

# Lazy import for optional Phase 2 components
_vector_store_class = None
_embedding_service_class = None

# Lazy import for Phase 4 consolidation
_consolidation_worker_class = None


def _load_consolidation_worker():
    """Lazy load Phase 4 consolidation worker"""
    global _consolidation_worker_class
    if _consolidation_worker_class is None:
        try:
            from .consolidation_worker import ConsolidationWorker
            _consolidation_worker_class = ConsolidationWorker
            return True
        except ImportError as e:
            logger.warning(f"Phase 4 consolidation worker not available: {e}")
            return False
    return True


def _load_phase2_components():
    """Lazy load Phase 2 components (vector store and embeddings)"""
    global _vector_store_class, _embedding_service_class
    if _vector_store_class is None:
        try:
            from .vector_store import VectorStore
            from .embedding_service import EmbeddingService
            _vector_store_class = VectorStore
            _embedding_service_class = EmbeddingService
            return True
        except ImportError as e:
            logger.warning(f"Phase 2 components not available: {e}")
            return False
    return True


class MemorySystem:
    """
    Complete memory system orchestrating all layers.
    
    Pipeline:
    1. EXTRACT - Identify what's worth remembering (Stages 1-3)
    2. STORE - Persist memories (flat files + Redis + Vector store)
    3. DEDUPLICATE - Check for semantic duplicates (Phase 3)
    4. RETRIEVE - Find relevant memories (semantic + type + recency)
    5. INJECT - Compose into prompt
    6. RESPOND - Generate response (external LLM call)
    
    Phase 2 adds:
    - Vector store (Qdrant) for semantic search
    - Embedding generation for memories
    - Multi-signal ranking combining semantic, type, and recency scores
    
    Phase 3 adds:
    - Stage 3 LLM extraction for complex cases
    - Semantic deduplication using vector similarity
    - Memory superseding and updates
    - Confidence boosting for repeated mentions
    """

    def __init__(self, user_id: str, enable_semantic_search: Optional[bool] = None, json_log_path: Optional[str] = None):
        """
        Initialize memory system for a user.
        
        Args:
            user_id: Unique user identifier
            enable_semantic_search: Override config to enable/disable semantic search
                                   (None = use config default)
            json_log_path: Optional path to JSON file for logging per-turn stats
                          (None = no JSON logging)
        """
        self.user_id = user_id
        self.turn_number = 0
        self.json_log_path = Path(json_log_path) if json_log_path else None
        self.turn_stats_log = []
        
        # Determine if semantic search should be enabled
        use_semantic = enable_semantic_search if enable_semantic_search is not None else SEMANTIC_SEARCH_ENABLED
        
        # Initialize storage layers
        self.flat_file_store = FlatFileStore(user_id)
        self.redis_store = RedisStore()
        
        # Initialize Phase 2 components if enabled
        self.vector_store = None
        self.embedding_service = None
        self._semantic_enabled = False
        
        if use_semantic and _load_phase2_components():
            try:
                self.embedding_service = _embedding_service_class()
                self.vector_store = _vector_store_class(embedding_service=self.embedding_service)
                self._semantic_enabled = True
                logger.info("Phase 2 semantic search enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Phase 2 components: {e}")
                logger.info("Falling back to Phase 1 mode (no semantic search)")
        
        # Initialize processing modules
        self.extractor = MemoryExtractor()
        self.retriever = MemoryRetriever(
            self.redis_store, 
            vector_store=self.vector_store if self._semantic_enabled else None,
            use_5_signal=True,  # Phase 4: Use 5-signal ranking
            user_id=self.user_id,  # Pass user_id for multi-user isolation
        )
        
        # Initialize Phase 4 consolidation worker
        self.consolidation_worker = None
        self._consolidation_enabled = False
        
        if CONSOLIDATION_ENABLED and _load_consolidation_worker():
            try:
                self.consolidation_worker = _consolidation_worker_class(
                    redis_store=self.redis_store,
                    flat_file_store=self.flat_file_store,
                    vector_store=self.vector_store,
                    embedding_service=self.embedding_service,
                )
                self._consolidation_enabled = True
                logger.info("Phase 4 consolidation worker enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize consolidation worker: {e}")
        
        logger.info(f"Initialized memory system for user {user_id} (semantic={self._semantic_enabled})")

    def process_turn(
        self,
        user_message: str,
        priority_types: Optional[List[str]] = None,
        speaker: Optional[str] = None,
        event_date: Optional[str] = None,
        event_ts: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        """
        Process a single conversation turn.

        This is the main entry point for the memory system.
        Call this for each user message before generating a response.

        Args:
            user_message: The user's message text
            priority_types: Optional list of memory types to prioritize
            speaker: Who is speaking. Required to keep several people's facts apart --
                without it every participant's memories merge into one pool and
                attribution is silently wrong. Optional so single-user callers are
                unaffected.
            event_date: Human-readable date the turn refers to (e.g. "8 May, 2023").
                Distinct from `timestamp`, which is ingest wall-clock: for replayed or
                backfilled conversations those differ by years, and only this one can
                answer "when did that happen".
            event_ts: Epoch form of event_date, for range and proximity comparisons.

        Returns:
            (memory_context, stats) where:
            - memory_context: Formatted string to inject into prompt
            - stats: Dictionary with extraction/retrieval statistics
        """
        self.turn_number += 1
        logger.info(f"\n{'='*60}\nProcessing turn {self.turn_number}\n{'='*60}")
        
        stats = {
            'turn_number': self.turn_number,
            'extracted_count': 0,
            'stored_count': 0,
            'retrieved_count': 0,
            'total_memories': 0,
            'extraction_time_ms': 0.0,
            'storage_time_ms': 0.0,
            'retrieval_time_ms': 0.0,
            'response_generated': True,
        }
        
        # STEP 1: EXTRACT - Identify memories from this message
        extract_start = time.time()
        extracted_memories = self.extractor.extract(
            user_message,
            self.turn_number,
            speaker=speaker,
            event_date=event_date,
            event_ts=event_ts,
        )

        # Add user_id to all extracted memories
        for memory in extracted_memories:
            memory['user_id'] = self.user_id
        
        stats['extraction_time_ms'] = (time.time() - extract_start) * 1000
        stats['extracted_count'] = len(extracted_memories)
        
        # STEP 2 & 3: STORE + DEDUPLICATE - Persist extracted memories with deduplication
        storage_start = time.time()
        stored_count = 0
        vector_stored_count = 0
        dedup_count = 0
        superseded_count = 0
        
        for memory in extracted_memories:
            # Phase 3: Semantic deduplication check
            should_store = True
            duplicate_id = None
            
            if SEMANTIC_DEDUP_ENABLED and self._semantic_enabled and self.vector_store:
                duplicate_id = self._check_semantic_duplicate(memory)
                
                if duplicate_id:
                    dedup_count += 1
                    should_store = False
                    
                    # If this is marked as an update, supersede the duplicate
                    if memory.get('is_update', False):
                        # Store the new memory
                        should_store = True
                        # Supersede the old one after storing
                        logger.info(f"Update detected: will supersede {duplicate_id}")
                    else:
                        # Just boost confidence of existing memory
                        self.redis_store.boost_confidence(duplicate_id)
                        logger.info(f"Duplicate found: boosted confidence of {duplicate_id}")
            
            # Store if not a duplicate or if it's an update
            if should_store:
                success = self.redis_store.store_memory(memory)
                if success:
                    stored_count += 1
                    
                    # Supersede old memory if this is an update
                    if duplicate_id and memory.get('is_update', False):
                        self.redis_store.supersede_memory(duplicate_id, memory['memory_id'])
                        superseded_count += 1
                    
                    # Phase 2: Also store in vector store for semantic search
                    if self._semantic_enabled and self.vector_store:
                        try:
                            self.vector_store.store_memory(memory)
                            vector_stored_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to store memory in vector store: {e}")
        
        stats['stored_count'] = stored_count
        stats['vector_stored_count'] = vector_stored_count
        stats['dedup_count'] = dedup_count
        stats['superseded_count'] = superseded_count
        stats['semantic_enabled'] = self._semantic_enabled
        stats['total_memories'] = self.redis_store.count_memories()
        stats['storage_time_ms'] = (time.time() - storage_start) * 1000
        
        # STEP 4 & 5: RETRIEVE + INJECT - Get relevant memories and format for prompt
        retrieval_start = time.time()
        memory_context, active_memories = self._compose_prompt_context(
            user_message, 
            priority_types
        )
        stats['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000
        
        # Count retrieved memories and expose active memories
        stats['retrieved_count'] = len(active_memories)
        stats['active_memories'] = active_memories
        
        # Phase 4: Check if consolidation should run
        if self._consolidation_enabled and self.consolidation_worker:
            if self.consolidation_worker.needs_consolidation(
                self.turn_number, 
                CONSOLIDATION_INTERVAL_TURNS
            ):
                logger.info("Running background consolidation...")
                consolidation_stats = self.consolidation_worker.run_consolidation(self.turn_number)
                stats['consolidation'] = consolidation_stats
        
        logger.info(
            f"Turn {self.turn_number} complete: "
            f"extracted={stats['extracted_count']}, "
            f"stored={stats['stored_count']}, "
            f"retrieved={stats['retrieved_count']}, "
            f"total={stats['total_memories']}"
        )
        
        # Save to JSON log if enabled
        if self.json_log_path:
            self._save_turn_stats_to_json(stats)
        
        return memory_context, stats
    
    def _check_semantic_duplicate(self, memory: Dict) -> Optional[str]:
        """
        Check if a memory is a semantic duplicate of existing memories.
        
        Args:
            memory: New memory to check
        
        Returns:
            memory_id of duplicate if found, None otherwise
        """
        try:
            similar = self.vector_store.find_similar_for_dedup(
                memory=memory,
                threshold=SEMANTIC_DEDUP_THRESHOLD,
                limit=SEMANTIC_DEDUP_CHECK_LIMIT
            )
            
            if similar:
                # Return the most similar memory's ID
                duplicate_id, score = similar[0]

                # Two recountings of the SAME KIND of event on DIFFERENT dates are two
                # events, not a duplicate. Merging them is not a harmless space saving:
                # it erases precisely the distinctions that "when did X happen" and
                # any chain of reasoning across sessions depend on. Cosine similarity
                # cannot see dates, so this has to be checked explicitly.
                if SEMANTIC_DEDUP_RESPECTS_DATE:
                    new_date = (memory.get('event_date') or '').strip()
                    if new_date:
                        existing = self.redis_store.get_memory(duplicate_id)
                        old_date = ((existing or {}).get('event_date') or '').strip()
                        if old_date and old_date != new_date:
                            logger.info(
                                f"Near-duplicate {duplicate_id} kept separate: "
                                f"different event dates ({old_date!r} vs {new_date!r})"
                            )
                            return None

                logger.info(
                    f"Semantic duplicate detected: {duplicate_id} "
                    f"(similarity={score:.3f}, threshold={SEMANTIC_DEDUP_THRESHOLD})"
                )
                return duplicate_id

            return None
            
        except Exception as e:
            logger.error(f"Semantic dedup check failed: {e}")
            return None

    def _compose_prompt_context(
        self, 
        user_message: str, 
        priority_types: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Compose the full memory context for prompt injection.
        
        Structure:
        1. Core Memory (ALWAYS injected)
        2. Retrieved Long-Term Memories (selective)
        
        Args:
            user_message: Current user message
            priority_types: Memory types to prioritize
        
        Returns:
            (formatted_context, active_memories) where:
            - formatted_context: Complete formatted memory context string
            - active_memories: List of memory dicts with metadata
        """
        sections = []
        active_memories = []
        
        # Layer 1: Core Memory (always injected)
        core_memory = self.flat_file_store.read_core_memory()
        if core_memory.strip():
            sections.append("### CORE MEMORY (Always Active)")
            sections.append(core_memory)
        
        # Layer 4: Long-Term Memory (selective retrieval)
        retrieved_memories_list = self.retriever.retrieve(
            user_message,
            self.turn_number,
            priority_types,
        )
        
        # Update access tracking for retrieved memories
        for mem in retrieved_memories_list:
            self.redis_store.increment_access_count(mem['memory_id'], self.turn_number)
        
        # Format the active memories for exposure
        for mem in retrieved_memories_list:
            active_memories.append({
                "memory_id": mem.get('memory_id', 'unknown'),
                "content": f"{mem.get('key', 'unknown')}: {mem.get('value', '')}",
                "type": mem.get('type', 'fact'),
                "origin_turn": mem.get('turn_number', 0),
                "last_used_turn": self.turn_number,
                "confidence": mem.get('confidence', 0),
                "mention_count": mem.get('mention_count', 1),
                "access_count": mem.get('access_count', 0) + 1  # Will be updated after
            })
        
        # Format memories for prompt
        retrieved_formatted = self.retriever.format_memories_for_prompt(retrieved_memories_list)
        
        if retrieved_formatted.strip():
            sections.append("### LONG-TERM MEMORY (Retrieved)")
            sections.append(retrieved_formatted)
        
        return "\n\n".join(sections), active_memories

    def get_prompt_context(
        self, 
        user_message: str,
        priority_types: Optional[List[str]] = None,
    ) -> str:
        """
        Get memory context without processing the turn.
        Useful for testing retrieval without extraction.
        
        Args:
            user_message: Message to retrieve memories for
            priority_types: Memory types to prioritize
        
        Returns:
            Formatted memory context
        """
        return self._compose_prompt_context(user_message, priority_types)

    def update_core_memory(self, file: str, section: str, field: str, value: str):
        """
        Update a field in core memory.
        Use sparingly - only for high-confidence identity updates.
        
        Args:
            file: Core file name (e.g., "CORE.md")
            section: Section within the file
            field: Field to update
            value: New value
        """
        self.flat_file_store.update_core_field(file, section, field, value)
        logger.info(f"Updated core memory: {file} -> {section} -> {field}")

    def get_statistics(self) -> Dict:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary with counts and metrics
        """
        stats = {
            'total_turns': self.turn_number,
            'total_memories': self.redis_store.count_memories(),
            'memories_by_type': {},
            'extraction_count': self.extractor.extraction_count,
            'semantic_search_enabled': self._semantic_enabled,
        }
        
        # Count by type
        from .config import MEMORY_TYPES
        for mem_type in MEMORY_TYPES:
            count = self.redis_store.count_memories_by_type(mem_type)
            if count > 0:
                stats['memories_by_type'][mem_type] = count
        
        # Phase 2: Add vector store stats
        if self._semantic_enabled and self.vector_store:
            try:
                stats['vector_count'] = self.vector_store.count()
            except Exception:
                stats['vector_count'] = -1
        
        return stats

    def clear_memories(self):
        """Clear all long-term memories (use with caution!)"""
        # Check if there are memories to clear
        all_mems = self.redis_store.get_all_memories()
        
        if not all_mems:
            logger.debug(f"No memories to clear for user {self.user_id}")
            return
        
        logger.warning(f"Clearing {len(all_mems)} memories for user {self.user_id}")
        self.redis_store.clear_all_memories()
        
        # Phase 2: Also clear vector store
        if self._semantic_enabled and self.vector_store:
            try:
                self.vector_store.clear()
                logger.info("Cleared vector store")
            except Exception as e:
                logger.warning(f"Failed to clear vector store: {e}")

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all storage layers.
        
        Returns:
            Dictionary with status of each component
        """
        health = {
            'redis': self.redis_store.health_check(),
            'flat_files': self.flat_file_store.user_dir.exists(),
        }
        
        # Phase 2: Check vector store health
        if self._semantic_enabled and self.vector_store:
            try:
                # Try to get count as health check
                self.vector_store.count()
                health['vector_store'] = True
            except Exception:
                health['vector_store'] = False
        else:
            health['vector_store'] = None  # Not enabled
        
        return health

    # =========================================================================
    # Phase 4: Consolidation Methods
    # =========================================================================
    
    def run_consolidation(self, force: bool = False) -> Optional[Dict]:
        """
        Manually trigger consolidation.
        
        Args:
            force: Force run even if not enough memories
        
        Returns:
            Consolidation statistics or None if not available
        """
        if not self._consolidation_enabled or not self.consolidation_worker:
            logger.warning("Consolidation worker not available")
            return None
        
        return self.consolidation_worker.run_consolidation(self.turn_number, force=force)

    def get_consolidation_stats(self) -> Optional[Dict]:
        """
        Get consolidation statistics.
        
        Returns:
            Consolidation stats or None if not available
        """
        if not self._consolidation_enabled or not self.consolidation_worker:
            return None
        
        return self.consolidation_worker.get_stats()

    def is_consolidation_enabled(self) -> bool:
        """Check if consolidation is enabled"""
        return self._consolidation_enabled

    # =========================================================================
    # JSON Logging Methods
    # =========================================================================
    
    def _save_turn_stats_to_json(self, stats: Dict):
        """
        Save turn statistics to JSON file.
        
        Args:
            stats: Statistics dictionary from process_turn()
        """
        if not self.json_log_path:
            return
        
        try:
            # Create a serializable copy of stats (remove non-JSON types)
            json_stats = {
                'turn_number': stats['turn_number'],
                'extracted_count': stats['extracted_count'],
                'stored_count': stats['stored_count'],
                'retrieved_count': stats['retrieved_count'],
                'total_memories': stats['total_memories'],
                'extraction_time_ms': stats['extraction_time_ms'],
                'storage_time_ms': stats['storage_time_ms'],
                'retrieval_time_ms': stats['retrieval_time_ms'],
                'response_generated': stats['response_generated'],
                'active_memories': stats.get('active_memories', []),
                # Phase 2
                'semantic_enabled': stats.get('semantic_enabled', False),
                'vector_stored_count': stats.get('vector_stored_count', 0),
                # Phase 3
                'dedup_count': stats.get('dedup_count', 0),
                'superseded_count': stats.get('superseded_count', 0),
                # Phase 4
                'consolidation': stats.get('consolidation', None)
            }
            
            # Append to log
            self.turn_stats_log.append(json_stats)
            
            # Write to file
            self.json_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'user_id': self.user_id,
                    'total_turns': self.turn_number,
                    'conversation_turns': self.turn_stats_log
                }, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved turn {self.turn_number} stats to {self.json_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save turn stats to JSON: {e}")
    
    def get_turn_stats_log(self) -> List[Dict]:
        """
        Get all logged turn statistics.
        
        Returns:
            List of turn statistics dictionaries
        """
        return self.turn_stats_log.copy()
    
    def export_turn_stats(self, output_path: str):
        """
        Export turn statistics to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_id': self.user_id,
                'total_turns': self.turn_number,
                'conversation_turns': self.turn_stats_log
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.turn_stats_log)} turn stats to {output_file}")
