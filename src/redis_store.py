"""
Redis Storage Layer - Long-Term Memory (Phase 3)
Handles structured key-value storage with indices and semantic deduplication
"""

import logging
import json
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import redis

from .config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_MEMORY_PREFIX,
    REDIS_DEDUP_PREFIX,
    REDIS_TYPE_INDEX_PREFIX,
    REDIS_RECENCY_INDEX,
    MEMORY_FIELDS,
    SEMANTIC_DEDUP_ENABLED,
    CONFIDENCE_BOOST_PER_MENTION,
    MAX_CONFIDENCE,
)

logger = logging.getLogger(__name__)


class RedisStore:
    """Manages Redis storage for long-term memories"""

    def __init__(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def store_memory(self, memory: Dict) -> bool:
        """
        Store a memory record in Redis with proper indexing.
        
        Args:
            memory: Dictionary with all memory fields
        
        Returns:
            True if stored successfully, False if duplicate
        """
        memory_id = memory['memory_id']
        memory_type = memory['type']
        memory_key = memory.get('key', '')
        
        # Check for key-based duplicates (Phase 1)
        dedup_key = f"{REDIS_DEDUP_PREFIX}{memory_type}:{memory_key}"
        existing = self.client.get(dedup_key)
        
        if existing:
            logger.debug(f"Key-based duplicate found: {dedup_key} -> {existing}")
            # Update existing memory's recency and boost confidence
            self._update_recency(existing, memory['timestamp'])
            self.boost_confidence(existing)
            return False
        
        # Ensure Phase 3 fields have defaults (Redis-compatible)
        memory.setdefault('mention_count', 1)
        memory.setdefault('superseded_by', '')  # Empty string instead of None
        memory.setdefault('supersedes', '')     # Empty string instead of None
        memory.setdefault('is_update', False)
        memory.setdefault('last_accessed_turn', memory.get('turn_number', 0))
        
        # Phase 4 fields
        memory.setdefault('access_count', 0)
        memory.setdefault('merged_from', '')
        memory.setdefault('promoted_to_core', False)
        memory.setdefault('decay_applied', 0.0)
        
        # Convert None values to empty strings for Redis compatibility
        memory_to_store = {}
        for key, value in memory.items():
            if value is None:
                memory_to_store[key] = ''
            elif isinstance(value, bool):
                # Convert boolean to string for Redis
                memory_to_store[key] = str(value)
            else:
                memory_to_store[key] = value
        
        # Store the full memory record as a hash
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        self.client.hset(redis_key, mapping=memory_to_store)
        
        # Create dedup entry
        if memory_key:
            self.client.set(dedup_key, memory_id)
        
        # Add to type-based index
        type_index_key = f"{REDIS_TYPE_INDEX_PREFIX}{memory_type}"
        self.client.sadd(type_index_key, memory_id)
        
        # Add to recency-ordered index (sorted set by timestamp)
        self.client.zadd(REDIS_RECENCY_INDEX, {memory_id: memory['timestamp']})
        
        logger.info(f"Stored memory {memory_id} (type={memory_type}, key={memory_key})")
        return True

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Unique memory identifier
        
        Returns:
            Memory dictionary or None if not found
        """
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        memory = self.client.hgetall(redis_key)
        
        if not memory:
            return None
        
        # Convert string fields back to appropriate types
        if 'confidence' in memory:
            memory['confidence'] = float(memory['confidence'])
        if 'turn_number' in memory:
            memory['turn_number'] = int(memory['turn_number'])
        if 'timestamp' in memory:
            memory['timestamp'] = float(memory['timestamp'])
        if 'mention_count' in memory:
            memory['mention_count'] = int(memory['mention_count'])
        if 'last_accessed_turn' in memory:
            memory['last_accessed_turn'] = int(memory['last_accessed_turn'])
        if 'is_update' in memory:
            # Convert string back to boolean
            memory['is_update'] = memory['is_update'] in ['True', 'true', '1', True]
        
        # Phase 4 field conversions
        if 'access_count' in memory:
            memory['access_count'] = int(memory['access_count'])
        if 'decay_applied' in memory:
            memory['decay_applied'] = float(memory['decay_applied'])
        if 'promoted_to_core' in memory:
            memory['promoted_to_core'] = memory['promoted_to_core'] in ['True', 'true', '1', True]
        
        # Convert empty strings back to None for Phase 3 fields
        if 'superseded_by' in memory and memory['superseded_by'] == '':
            memory['superseded_by'] = None
        if 'supersedes' in memory and memory['supersedes'] == '':
            memory['supersedes'] = None
        
        return memory

    def get_memories_by_type(self, memory_type: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve all memories of a specific type.
        
        Args:
            memory_type: Type of memory (preference, constraint, etc.)
            limit: Maximum number to retrieve
        
        Returns:
            List of memory dictionaries
        """
        type_index_key = f"{REDIS_TYPE_INDEX_PREFIX}{memory_type}"
        memory_ids = self.client.smembers(type_index_key)
        
        memories = []
        for memory_id in list(memory_ids)[:limit]:
            memory = self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories

    def get_recent_memories(self, limit: int = 20) -> List[Dict]:
        """
        Retrieve the N most recent memories.
        
        Args:
            limit: Maximum number to retrieve
        
        Returns:
            List of memory dictionaries, sorted by recency (newest first)
        """
        # Get most recent memory IDs from sorted set (reverse order)
        memory_ids = self.client.zrevrange(REDIS_RECENCY_INDEX, 0, limit - 1)
        
        memories = []
        for memory_id in memory_ids:
            memory = self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories

    def _update_recency(self, memory_id: str, timestamp: float):
        """Update the timestamp of a memory in the recency index"""
        self.client.zadd(REDIS_RECENCY_INDEX, {memory_id: timestamp})

    def count_memories(self) -> int:
        """Count total number of stored memories"""
        return self.client.zcard(REDIS_RECENCY_INDEX)

    def count_memories_by_type(self, memory_type: str) -> int:
        """Count memories of a specific type"""
        type_index_key = f"{REDIS_TYPE_INDEX_PREFIX}{memory_type}"
        return self.client.scard(type_index_key)

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory and all its indices.
        
        Args:
            memory_id: Memory to delete
        
        Returns:
            True if deleted, False if not found
        """
        # Get memory to find its type and key
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        
        # Delete from main storage
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        self.client.delete(redis_key)
        
        # Delete from dedup index
        if memory.get('key'):
            dedup_key = f"{REDIS_DEDUP_PREFIX}{memory['type']}:{memory['key']}"
            self.client.delete(dedup_key)
        
        # Remove from type index
        type_index_key = f"{REDIS_TYPE_INDEX_PREFIX}{memory['type']}"
        self.client.srem(type_index_key, memory_id)
        
        # Remove from recency index
        self.client.zrem(REDIS_RECENCY_INDEX, memory_id)
        
        logger.info(f"Deleted memory {memory_id}")
        return True
    
    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        """
        Mark an old memory as superseded by a new one.
        
        Args:
            old_memory_id: Memory to be superseded
            new_memory_id: Memory that supersedes the old one
        
        Returns:
            True if successful
        """
        old_memory = self.get_memory(old_memory_id)
        new_memory = self.get_memory(new_memory_id)
        
        if not old_memory or not new_memory:
            logger.warning("Can't supersede: memory not found")
            return False
        
        # Update old memory (use actual memory_id, not empty string)
        old_redis_key = f"{REDIS_MEMORY_PREFIX}{old_memory_id}"
        self.client.hset(old_redis_key, "superseded_by", new_memory_id)
        
        # Update new memory (use actual memory_id, not empty string)
        new_redis_key = f"{REDIS_MEMORY_PREFIX}{new_memory_id}"
        self.client.hset(new_redis_key, "supersedes", old_memory_id)
        
        logger.info(f"Memory {old_memory_id} superseded by {new_memory_id}")
        return True
    
    def boost_confidence(self, memory_id: str) -> bool:
        """
        Boost confidence of a memory when it's mentioned again.
        
        Args:
            memory_id: Memory to boost
        
        Returns:
            True if successful
        """
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        
        # Increment mention count
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        mention_count = int(memory.get('mention_count', 1)) + 1
        self.client.hset(redis_key, "mention_count", mention_count)
        
        # Boost confidence
        current_confidence = float(memory.get('confidence', 0.5))
        new_confidence = min(current_confidence + CONFIDENCE_BOOST_PER_MENTION, MAX_CONFIDENCE)
        self.client.hset(redis_key, "confidence", new_confidence)
        
        logger.info(
            f"Boosted memory {memory_id}: mentions={mention_count}, "
            f"confidence={current_confidence:.2f}→{new_confidence:.2f}"
        )
        return True
    
    def update_memory_field(self, memory_id: str, field: str, value: any) -> bool:
        """
        Update a specific field of a memory.
        
        Args:
            memory_id: Memory to update
            field: Field name
            value: New value
        
        Returns:
            True if successful
        """
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        self.client.hset(redis_key, field, value)
        
        logger.debug(f"Updated memory {memory_id}: {field}={value}")
        return True
    
    def get_memories_by_type_and_key(self, memory_type: str, key: str) -> List[Dict]:
        """
        Find memories with specific type and key.
        
        Args:
            memory_type: Memory type
            key: Memory key
        
        Returns:
            List of matching memories
        """
        all_of_type = self.get_memories_by_type(memory_type, limit=1000)
        return [m for m in all_of_type if m.get('key') == key]

    def clear_all_memories(self):
        """Clear all memories (use with caution!)"""
        # Get all memory IDs first to check if there's anything to clear
        memory_ids = self.client.zrange(REDIS_RECENCY_INDEX, 0, -1)
        
        if not memory_ids:
            logger.debug("No memories to clear (Redis already empty)")
            return
        
        logger.warning(f"Clearing {len(memory_ids)} memories from Redis")
        
        # Delete each memory properly
        for memory_id in memory_ids:
            self.delete_memory(memory_id)

    def increment_access_count(self, memory_id: str, current_turn: int) -> bool:
        """
        Increment access count for a memory (Phase 4).
        Called when a memory is retrieved for prompt injection.
        
        Args:
            memory_id: Memory to track
            current_turn: Current conversation turn
        
        Returns:
            True if successful
        """
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        
        redis_key = f"{REDIS_MEMORY_PREFIX}{memory_id}"
        
        # Increment access count
        current_count = int(memory.get('access_count', 0))
        self.client.hset(redis_key, "access_count", current_count + 1)
        
        # Update last accessed turn
        self.client.hset(redis_key, "last_accessed_turn", current_turn)
        
        logger.debug(f"Memory {memory_id} accessed (count={current_count + 1})")
        return True

    def get_all_memories(self, limit: int = 10000) -> List[Dict]:
        """
        Get all memories (for consolidation).
        
        Args:
            limit: Maximum number to retrieve
        
        Returns:
            List of all memory dictionaries
        """
        memory_ids = self.client.zrange(REDIS_RECENCY_INDEX, 0, limit - 1)
        memories = []
        for memory_id in memory_ids:
            memory = self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        return memories

    def health_check(self) -> bool:
        """Check if Redis is responsive"""
        try:
            return self.client.ping()
        except:
            return False
