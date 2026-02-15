"""
Vector Store - Phase 2
Qdrant-based vector storage for semantic similarity search
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant-based vector store for semantic memory search.
    
    Stores memory embeddings and enables:
    - Semantic similarity search
    - Filtered search by memory type
    - Hybrid search with multiple signals
    """
    
    def __init__(
        self, 
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_service: Service for generating embeddings (uses singleton if not provided)
            collection_name: Name of the Qdrant collection (uses default from config if not provided)
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        
        try:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
            logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection(self):
        """Create the collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                )
                
                # Create payload index for memory_type for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="memory_type",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def store_memory(self, memory: Dict, embedding: Optional[List[float]] = None) -> bool:
        """
        Store a memory with its embedding in the vector store.
        
        Args:
            memory: Memory dictionary with 'memory_id', 'type', 'key', 'value', etc.
            embedding: Pre-computed embedding (computed if not provided)
        
        Returns:
            True if stored successfully
        """
        memory_id = memory['memory_id']
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_service.embed_memory(memory)
        
        try:
            # Prepare payload (metadata stored with vector)
            payload = {
                "memory_id": memory_id,
                "user_id": memory.get('user_id', 'unknown'),  # Multi-user support
                "memory_type": memory.get('type', 'fact'),
                "key": memory.get('key', ''),
                "value": memory.get('value', ''),
                "confidence": float(memory.get('confidence', 0.5)),
                "turn_number": int(memory.get('turn_number', 0)),
                "timestamp": float(memory.get('timestamp', 0)),
            }
            
            # Upsert point (update if exists, insert if new)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=self._hash_id(memory_id),
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
            
            logger.debug(f"Stored memory {memory_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            return False
    
    def store_memories_batch(
        self, 
        memories: List[Dict], 
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """
        Store multiple memories in batch (more efficient).
        
        Args:
            memories: List of memory dictionaries
            embeddings: Pre-computed embeddings (computed if not provided)
        
        Returns:
            Number of memories stored successfully
        """
        if not memories:
            return 0
        
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = [self.embedding_service.embed_memory(m) for m in memories]
        
        try:
            points = []
            for memory, embedding in zip(memories, embeddings):
                payload = {
                    "memory_id": memory['memory_id'],
                    "memory_type": memory.get('type', 'fact'),
                    "key": memory.get('key', ''),
                    "value": memory.get('value', ''),
                    "confidence": float(memory.get('confidence', 0.5)),
                    "turn_number": int(memory.get('turn_number', 0)),
                    "timestamp": float(memory.get('timestamp', 0)),
                }
                
                points.append(
                    models.PointStruct(
                        id=self._hash_id(memory['memory_id']),
                        vector=embedding,
                        payload=payload,
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            logger.info(f"Stored {len(points)} memories in vector store (batch)")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to store memories batch: {e}")
            return 0
    
    def search_similar(
        self, 
        query: str, 
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,  # Filter by user
    ) -> List[Dict]:
        """
        Search for memories similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            memory_types: Filter by these memory types (None for all types)
            min_score: Minimum similarity score threshold
            user_id: Filter by user ID (None for all users)
        
        Returns:
            List of dictionaries with 'memory', 'score', 'memory_id' keys
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        return self.search_by_vector(
            query_embedding, 
            limit=limit, 
            memory_types=memory_types,
            min_score=min_score,
            user_id=user_id,
        )
    
    def search_by_vector(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,  # Filter by user
    ) -> List[Dict]:
        """
        Search for memories using a pre-computed query vector.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            memory_types: Filter by these memory types (None for all types)
            min_score: Minimum similarity score threshold
            user_id: Filter by user ID (None for all users)
        
        Returns:
            List of dictionaries with 'memory', 'score', 'memory_id' keys
        """
        try:
            # Build filters
            filter_conditions = []
            
            # Filter by user_id (critical for multi-user isolation)
            if user_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                )
            
            # Filter by memory types
            if memory_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchAny(any=memory_types),
                    )
                )
            
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(must=filter_conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=min_score,
            )
            
            # Convert to our format
            memories = []
            for result in results:
                memory = {
                    "memory_id": result.payload.get("memory_id"),
                    "type": result.payload.get("memory_type"),
                    "key": result.payload.get("key"),
                    "value": result.payload.get("value"),
                    "confidence": result.payload.get("confidence"),
                    "turn_number": result.payload.get("turn_number"),
                    "timestamp": result.payload.get("timestamp"),
                }
                
                memories.append({
                    "memory": memory,
                    "score": result.score,
                    "memory_id": result.payload.get("memory_id"),
                })
            
            logger.debug(f"Vector search found {len(memories)} results")
            return memories
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the vector store.
        
        Args:
            memory_id: Memory identifier to delete
        
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[self._hash_id(memory_id)],
                ),
            )
            logger.debug(f"Deleted memory {memory_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def count(self) -> int:
        """
        Get total number of vectors in the store.
        
        Returns:
            Count of stored vectors
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0
    
    def find_similar_for_dedup(
        self,
        memory: Dict,
        threshold: float = 0.92,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar memories for deduplication check.
        
        Args:
            memory: New memory to check for duplicates
            threshold: Minimum similarity score to consider duplicate
            limit: Maximum number of candidates to return
        
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        # Generate embedding for the new memory
        embedding = self.embedding_service.embed_memory(memory)
        
        # Search for similar memories of the same type AND same user
        memory_type = memory.get('type')
        memory_types = [memory_type] if memory_type else None
        user_id = memory.get('user_id')  # Extract user_id from memory
        
        results = self.search_by_vector(
            query_vector=embedding,
            limit=limit,
            memory_types=memory_types,
            min_score=threshold,
            user_id=user_id,  # Filter by user for multi-user isolation
        )
        
        # Return simplified list
        similar = [(r['memory_id'], r['score']) for r in results]
        
        if similar:
            logger.info(f"Found {len(similar)} similar memories for dedup check")
        
        return similar
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory from the vector store.
        
        Args:
            memory_id: ID of memory to delete
        
        Returns:
            True if deleted successfully
        """
        try:
            point_id = self._hash_id(memory_id)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id],
                ),
            )
            logger.debug(f"Deleted memory {memory_id} from vector store")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id} from vector store: {e}")
            return False
    
    def add_memory(self, memory: Dict) -> bool:
        """
        Alias for store_memory for consistency.
        
        Args:
            memory: Memory dictionary to store
        
        Returns:
            True if stored successfully
        """
        return self.store_memory(memory)
    
    def clear(self) -> bool:
        """
        Clear all vectors from the collection (use with caution).
        
        Returns:
            True if cleared successfully
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared vector store collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            return False
    
    def _hash_id(self, memory_id: str) -> int:
        """
        Convert string memory_id to integer for Qdrant.
        
        Qdrant requires integer IDs, so we hash the string ID.
        Uses a stable hash to ensure consistency.
        """
        # Use a simple hash that's stable across runs
        import hashlib
        hash_bytes = hashlib.md5(memory_id.encode()).digest()
        # Take first 8 bytes as unsigned int
        return int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get the singleton vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
