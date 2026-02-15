"""
Phase 4: Consolidation Worker
Handles background memory consolidation including:
- Memory decay (reduce confidence of old/unused memories)
- Memory merging (combine similar memories)
- Core Memory promotion (promote high-value memories to flat files)
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import (
    # Decay configuration
    MEMORY_DECAY_ENABLED,
    DECAY_TURNS_THRESHOLD,
    DECAY_INACTIVE_TURNS,
    DECAY_RATE_PER_100_TURNS,
    MIN_DECAY_CONFIDENCE,
    DELETE_VERY_LOW_CONFIDENCE,
    # Merge configuration
    MEMORY_MERGE_ENABLED,
    MERGE_SIMILARITY_THRESHOLD,
    MERGE_SAME_TYPE_ONLY,
    MAX_MERGED_VALUE_LENGTH,
    # Promotion configuration
    PROMOTION_ENABLED,
    PROMOTION_CONFIDENCE_THRESHOLD,
    PROMOTION_MENTION_THRESHOLD,
    PROMOTION_ACCESS_THRESHOLD,
    PROMOTION_AGE_THRESHOLD,
    PROMOTABLE_TYPES,
    # General
    CONSOLIDATION_MIN_MEMORIES,
    MAX_CONFIDENCE,
)
from .redis_store import RedisStore
from .flat_file_store import FlatFileStore

logger = logging.getLogger(__name__)


class ConsolidationWorker:
    """
    Background worker for memory consolidation.
    
    Performs three main operations:
    1. DECAY - Reduce confidence of old/unused memories
    2. MERGE - Combine semantically similar memories
    3. PROMOTE - Move high-value memories to Core Memory
    """

    def __init__(
        self, 
        redis_store: RedisStore, 
        flat_file_store: FlatFileStore,
        vector_store=None,
        embedding_service=None,
    ):
        """
        Initialize the consolidation worker.
        
        Args:
            redis_store: Redis store instance
            flat_file_store: Flat file store for core memory
            vector_store: Optional vector store for similarity checks
            embedding_service: Optional embedding service for merging
        """
        self.redis_store = redis_store
        self.flat_file_store = flat_file_store
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        self._last_consolidation_turn = 0
        self._consolidation_stats = {
            "decayed": 0,
            "deleted": 0,
            "merged": 0,
            "promoted": 0,
            "last_run": None,
        }

    def run_consolidation(self, current_turn: int, force: bool = False) -> Dict:
        """
        Run full consolidation cycle.
        
        Args:
            current_turn: Current conversation turn number
            force: Force run even if not enough memories
        
        Returns:
            Statistics about consolidation operations
        """
        start_time = time.time()
        stats = {
            "decayed": 0,
            "deleted": 0,
            "merged": 0,
            "promoted": 0,
            "skipped": False,
            "duration_ms": 0,
        }
        
        # Check minimum memories threshold
        total_memories = self.redis_store.count_memories()
        if total_memories < CONSOLIDATION_MIN_MEMORIES and not force:
            logger.info(f"Skipping consolidation: only {total_memories} memories")
            stats["skipped"] = True
            return stats
        
        logger.info(f"Starting consolidation at turn {current_turn} ({total_memories} memories)")
        
        # Step 1: Apply decay to old memories
        if MEMORY_DECAY_ENABLED:
            decay_stats = self._apply_decay(current_turn)
            stats["decayed"] = decay_stats["decayed"]
            stats["deleted"] = decay_stats["deleted"]
        
        # Step 2: Merge similar memories
        if MEMORY_MERGE_ENABLED and self.vector_store:
            merge_stats = self._merge_similar_memories(current_turn)
            stats["merged"] = merge_stats["merged"]
        
        # Step 3: Promote high-value memories to core
        if PROMOTION_ENABLED:
            promotion_stats = self._promote_to_core(current_turn)
            stats["promoted"] = promotion_stats["promoted"]
        
        # Update tracking
        self._last_consolidation_turn = current_turn
        stats["duration_ms"] = (time.time() - start_time) * 1000
        
        self._consolidation_stats = {
            **stats,
            "last_run": datetime.now().isoformat(),
        }
        
        logger.info(
            f"Consolidation complete: decayed={stats['decayed']}, "
            f"deleted={stats['deleted']}, merged={stats['merged']}, "
            f"promoted={stats['promoted']} ({stats['duration_ms']:.1f}ms)"
        )
        
        return stats

    def _apply_decay(self, current_turn: int) -> Dict:
        """
        Apply decay to old and unused memories.
        
        Decay formula:
        - Base decay: confidence -= (turns_old / 100) * DECAY_RATE
        - Inactive bonus: extra decay if not accessed recently
        
        Returns:
            Stats about decay operations
        """
        stats = {"decayed": 0, "deleted": 0}
        
        # Get all memories
        all_memories = self.redis_store.get_recent_memories(limit=10000)
        
        for memory in all_memories:
            memory_id = memory['memory_id']
            turn_created = int(memory.get('turn_number', 0))
            last_accessed = int(memory.get('last_accessed_turn', turn_created))
            current_confidence = float(memory.get('confidence', 0.5))
            decay_applied = float(memory.get('decay_applied', 0))
            
            # Skip if already promoted to core
            if memory.get('promoted_to_core') in ['True', 'true', True]:
                continue
            
            turns_old = current_turn - turn_created
            turns_since_access = current_turn - last_accessed
            
            # Check if decay should apply
            if turns_old < DECAY_TURNS_THRESHOLD:
                continue
            
            # Calculate decay amount
            decay_amount = (turns_old / 100) * DECAY_RATE_PER_100_TURNS
            
            # Extra decay for inactive memories
            if turns_since_access > DECAY_INACTIVE_TURNS:
                inactive_factor = (turns_since_access - DECAY_INACTIVE_TURNS) / 100
                decay_amount += inactive_factor * DECAY_RATE_PER_100_TURNS * 0.5
            
            # Apply decay
            new_confidence = max(0.1, current_confidence - decay_amount)
            total_decay = decay_applied + decay_amount
            
            # Check if should delete
            if new_confidence < MIN_DECAY_CONFIDENCE and DELETE_VERY_LOW_CONFIDENCE:
                logger.info(
                    f"Deleting low-confidence memory {memory_id} "
                    f"(confidence={new_confidence:.2f})"
                )
                self.redis_store.delete_memory(memory_id)
                
                # Also remove from vector store
                if self.vector_store:
                    try:
                        self.vector_store.delete_memory(memory_id)
                    except:
                        pass
                
                stats["deleted"] += 1
            else:
                # Update confidence and decay tracking
                self.redis_store.update_memory_field(memory_id, "confidence", new_confidence)
                self.redis_store.update_memory_field(memory_id, "decay_applied", total_decay)
                
                if decay_amount > 0.01:  # Log if meaningful decay
                    logger.debug(
                        f"Decayed {memory_id}: {current_confidence:.2f} → {new_confidence:.2f}"
                    )
                    stats["decayed"] += 1
        
        return stats

    def _merge_similar_memories(self, current_turn: int) -> Dict:
        """
        Merge semantically similar memories.
        
        Strategy:
        1. Find memory pairs with high similarity
        2. Check if same type (if configured)
        3. Create merged memory with combined value
        4. Mark old memories as superseded
        
        Returns:
            Stats about merge operations
        """
        stats = {"merged": 0, "candidates": 0}
        
        if not self.vector_store or not self.embedding_service:
            logger.warning("Vector store required for memory merging")
            return stats
        
        # Get all memories
        all_memories = self.redis_store.get_recent_memories(limit=1000)
        
        # Track which memories have been merged
        merged_ids = set()
        
        for memory in all_memories:
            memory_id = memory['memory_id']
            
            # Skip if already merged or superseded
            if memory_id in merged_ids:
                continue
            if memory.get('superseded_by'):
                continue
            if memory.get('promoted_to_core') in ['True', 'true', True]:
                continue
            
            # Find similar memories
            try:
                similar = self.vector_store.search_similar(
                    query=memory.get('value', ''),
                    limit=5,
                    min_score=MERGE_SIMILARITY_THRESHOLD,
                )
            except Exception as e:
                logger.debug(f"Similarity search failed: {e}")
                continue
            
            for result in similar:
                other_id = result['memory_id']
                similarity = result['score']
                
                # Skip self
                if other_id == memory_id:
                    continue
                
                # Skip if already merged
                if other_id in merged_ids:
                    continue
                
                # Get full other memory
                other_memory = self.redis_store.get_memory(other_id)
                if not other_memory:
                    continue
                
                # Check type matching
                if MERGE_SAME_TYPE_ONLY and memory['type'] != other_memory.get('type'):
                    continue
                
                # Skip if other is superseded
                if other_memory.get('superseded_by'):
                    continue
                
                # Perform merge
                merged_memory = self._create_merged_memory(
                    memory, other_memory, current_turn, similarity
                )
                
                if merged_memory:
                    # Store merged memory
                    stored = self.redis_store.store_memory(merged_memory)
                    
                    if stored:
                        # Mark old memories as superseded
                        self.redis_store.supersede_memory(memory_id, merged_memory['memory_id'])
                        self.redis_store.supersede_memory(other_id, merged_memory['memory_id'])
                        
                        # Add to vector store
                        if self.vector_store:
                            self.vector_store.add_memory(merged_memory)
                        
                        merged_ids.add(memory_id)
                        merged_ids.add(other_id)
                        stats["merged"] += 1
                        
                        logger.info(
                            f"Merged memories {memory_id} + {other_id} → "
                            f"{merged_memory['memory_id']} (similarity={similarity:.2f})"
                        )
                        break  # Move to next memory after successful merge
        
        return stats

    def _create_merged_memory(
        self, 
        mem1: Dict, 
        mem2: Dict, 
        current_turn: int,
        similarity: float,
    ) -> Optional[Dict]:
        """
        Create a merged memory from two similar memories.
        
        Strategy:
        - Use higher confidence memory's key
        - Combine values intelligently
        - Sum mention counts
        - Average timestamps
        """
        import hashlib
        
        # Determine primary memory (higher confidence)
        if float(mem1.get('confidence', 0)) >= float(mem2.get('confidence', 0)):
            primary, secondary = mem1, mem2
        else:
            primary, secondary = mem2, mem1
        
        # Combine values
        val1 = primary.get('value', '')
        val2 = secondary.get('value', '')
        
        if val1 == val2:
            merged_value = val1
        elif len(val1) > len(val2) and val2 in val1:
            merged_value = val1
        elif len(val2) > len(val1) and val1 in val2:
            merged_value = val2
        else:
            # Combine values
            merged_value = f"{val1}; {val2}"
            if len(merged_value) > MAX_MERGED_VALUE_LENGTH:
                merged_value = merged_value[:MAX_MERGED_VALUE_LENGTH]
        
        # Generate new ID
        merge_hash = hashlib.md5(
            f"{mem1['memory_id']}_{mem2['memory_id']}_{current_turn}".encode()
        ).hexdigest()[:8]
        new_id = f"merged_{merge_hash}"
        
        # Calculate merged confidence
        conf1 = float(primary.get('confidence', 0.5))
        conf2 = float(secondary.get('confidence', 0.5))
        merged_confidence = min(MAX_CONFIDENCE, (conf1 + conf2) / 2 + 0.05)
        
        # Sum mention counts
        mentions1 = int(primary.get('mention_count', 1))
        mentions2 = int(secondary.get('mention_count', 1))
        
        merged_memory = {
            'memory_id': new_id,
            'type': primary['type'],
            'key': primary.get('key', ''),
            'value': merged_value,
            'confidence': merged_confidence,
            'turn_number': current_turn,
            'timestamp': time.time(),
            'source_text': f"Merged from: {mem1['memory_id']}, {mem2['memory_id']}",
            'mention_count': mentions1 + mentions2,
            'superseded_by': '',
            'supersedes': '',
            'is_update': False,
            'last_accessed_turn': current_turn,
            'access_count': int(primary.get('access_count', 0)) + int(secondary.get('access_count', 0)),
            'merged_from': json.dumps([mem1['memory_id'], mem2['memory_id']]),
            'promoted_to_core': False,
            'decay_applied': 0,
        }
        
        return merged_memory

    def _promote_to_core(self, current_turn: int) -> Dict:
        """
        Promote high-value memories to Core Memory (flat files).
        
        Criteria for promotion:
        - High confidence (>= threshold)
        - Multiple mentions (>= threshold)
        - Frequently accessed (>= threshold)
        - Old enough (>= threshold turns)
        - Is a promotable type
        
        Returns:
            Stats about promotion operations
        """
        stats = {"promoted": 0, "candidates": 0}
        
        # Get all memories
        all_memories = self.redis_store.get_recent_memories(limit=10000)
        
        for memory in all_memories:
            memory_id = memory['memory_id']
            
            # Skip if already promoted
            if memory.get('promoted_to_core') in ['True', 'true', True]:
                continue
            
            # Skip if superseded
            if memory.get('superseded_by'):
                continue
            
            # Check promotion criteria
            confidence = float(memory.get('confidence', 0))
            mentions = int(memory.get('mention_count', 1))
            accesses = int(memory.get('access_count', 0))
            turns_old = current_turn - int(memory.get('turn_number', current_turn))
            mem_type = memory.get('type', '')
            
            # Check all thresholds
            if confidence < PROMOTION_CONFIDENCE_THRESHOLD:
                continue
            if mentions < PROMOTION_MENTION_THRESHOLD:
                continue
            if accesses < PROMOTION_ACCESS_THRESHOLD:
                continue
            if turns_old < PROMOTION_AGE_THRESHOLD:
                continue
            if mem_type not in PROMOTABLE_TYPES:
                continue
            
            stats["candidates"] += 1
            
            # Determine target file and section based on type
            target_file, section = self._get_promotion_target(mem_type)
            if not target_file:
                continue
            
            # Create promotion content
            key = memory.get('key', 'unknown')
            value = memory.get('value', '')
            content = f"**{key}:** {value}"
            
            # Add to core memory
            try:
                self.flat_file_store.append_to_section(target_file, section, content)
                
                # Mark as promoted
                self.redis_store.update_memory_field(memory_id, "promoted_to_core", "True")
                
                stats["promoted"] += 1
                logger.info(
                    f"Promoted memory {memory_id} to {target_file}/{section}: "
                    f"{key}={value[:50]}..."
                )
            except Exception as e:
                logger.error(f"Failed to promote {memory_id}: {e}")
        
        return stats

    def _get_promotion_target(self, mem_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine target file and section for promotion.
        
        Args:
            mem_type: Memory type
        
        Returns:
            Tuple of (filename, section_name) or (None, None)
        """
        promotion_mapping = {
            "entity": ("CORE.md", "Identity"),
            "preference": ("PREFERENCES.md", "General Preferences"),
            "constraint": ("CONSTRAINTS.md", "Hard Constraints"),
            "instruction": ("INSTRUCTIONS.md", "Communication Style"),
        }
        
        return promotion_mapping.get(mem_type, (None, None))

    def get_stats(self) -> Dict:
        """Get consolidation statistics"""
        return self._consolidation_stats.copy()

    def needs_consolidation(self, current_turn: int, interval: int) -> bool:
        """
        Check if consolidation should run.
        
        Args:
            current_turn: Current turn number
            interval: Number of turns between consolidation runs
        
        Returns:
            True if consolidation should run
        """
        if self._last_consolidation_turn == 0:
            # First consolidation after minimum memories
            return self.redis_store.count_memories() >= CONSOLIDATION_MIN_MEMORIES
        
        return (current_turn - self._last_consolidation_turn) >= interval
