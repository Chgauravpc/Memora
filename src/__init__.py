"""
Long-Form Memory System - Phase 1 & Phase 2
"""

from .memory_system import MemorySystem
from .flat_file_store import FlatFileStore
from .redis_store import RedisStore
from .extractor import MemoryExtractor
from .retriever import MemoryRetriever

# Phase 2 components (optional - may not be available if dependencies not installed)
try:
    from .embedding_service import EmbeddingService, get_embedding_service
    from .vector_store import VectorStore, get_vector_store
    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False
    EmbeddingService = None
    VectorStore = None
    get_embedding_service = None
    get_vector_store = None

__all__ = [
    'MemorySystem',
    'FlatFileStore',
    'RedisStore',
    'MemoryExtractor',
    'MemoryRetriever',
    # Phase 2
    'EmbeddingService',
    'VectorStore',
    'get_embedding_service',
    'get_vector_store',
    '_PHASE2_AVAILABLE',
]

__version__ = '0.2.0'
