"""
Embedding Service - Phase 2
Generates vector embeddings for memories and queries using sentence-transformers
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Lazy loading of the model to avoid slow startup
_model = None


def _get_model():
    """Lazy load the embedding model"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2 is fast and good quality for semantic search
            # Produces 384-dimensional embeddings
            model_name = "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _model


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model which produces 384-dimensional vectors.
    Good balance of quality and speed for semantic search use cases.
    """
    
    EMBEDDING_DIMENSION = 384  # Dimension of all-MiniLM-L6-v2
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    def __init__(self):
        """Initialize the embedding service (lazy loads model on first use)"""
        self._model = None
    
    @property
    def model(self):
        """Get the model, loading it if necessary"""
        if self._model is None:
            self._model = _get_model()
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.EMBEDDING_DIMENSION
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts and track their indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        # Create result array with zero vectors
        results = [[0.0] * self.EMBEDDING_DIMENSION for _ in range(len(texts))]
        
        if valid_texts:
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            for i, idx in enumerate(valid_indices):
                results[idx] = embeddings[i].tolist()
        
        return results
    
    def embed_memory(self, memory: dict) -> List[float]:
        """
        Generate embedding for a memory record.
        
        Combines key and value for richer semantic representation.
        
        Args:
            memory: Memory dictionary with 'key', 'value', 'type' fields
        
        Returns:
            Embedding vector
        """
        # Combine memory fields for embedding
        parts = []
        
        if memory.get('key'):
            parts.append(memory['key'])
        
        if memory.get('value'):
            parts.append(memory['value'])
        
        # Include type for additional context
        if memory.get('type'):
            parts.append(f"type: {memory['type']}")
        
        text = " | ".join(parts) if parts else ""
        return self.embed_text(text)
    
    def compute_similarity(
        self, 
        query_embedding: List[float], 
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query vector
            document_embeddings: List of document vectors
        
        Returns:
            List of similarity scores (0 to 1, higher is more similar)
        """
        if not document_embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(document_embeddings)
        
        # Normalize vectors
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        doc_vecs = doc_vecs / doc_norms
        
        # Compute cosine similarity
        similarities = np.dot(doc_vecs, query_vec)
        
        # Ensure values are between 0 and 1
        similarities = np.clip(similarities, 0, 1)
        
        return similarities.tolist()


# Singleton instance for convenience
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
