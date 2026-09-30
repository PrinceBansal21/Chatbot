"""
Abstract interface for retrieval strategies.

This interface defines the contract for different retrieval approaches:
- Standard: Basic vector similarity search
- Hierarchical: Parent-child chunk expansion
- Hybrid: Combined vector + keyword search
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result with content and metadata.
    
    Attributes:
        content: The text content of the retrieved chunk
        score: Similarity/relevance score (higher is better)
        metadata: Additional metadata (source, chunk_type, parent_id, etc.)
        chunk_id: Unique identifier for the chunk
        parent_content: For hierarchical retrieval, the parent chunk content
    """
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    parent_content: Optional[str] = None


@dataclass
class RetrievalQuery:
    """
    Encapsulates a retrieval query with all parameters.
    
    Attributes:
        query_text: The user's query string
        query_vector: Pre-computed query embedding (optional)
        top_k: Number of results to return
        filters: Metadata filters to apply
        include_parent: For hierarchical, whether to fetch parent chunks
        hybrid_alpha: For hybrid search, balance between vector (1.0) and keyword (0.0)
    """
    query_text: str
    query_vector: Optional[List[float]] = None
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    include_parent: bool = True
    hybrid_alpha: float = 0.7  # 0.7 = 70% vector, 30% keyword


class RetrieverManager(ABC):
    """
    Abstract interface for retrieval strategies.
    
    Implementations should handle:
    - Vector similarity search
    - Optional metadata filtering
    - Strategy-specific logic (parent expansion, hybrid scoring, etc.)
    """

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks based on the query.
        
        Args:
            query: RetrievalQuery containing query text/vector and parameters
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (descending)
        """
        pass

    @abstractmethod
    def retrieve_by_ids(self, chunk_ids: List[str]) -> List[RetrievalResult]:
        """
        Retrieve specific chunks by their IDs.
        
        Useful for fetching parent chunks in hierarchical retrieval.
        
        Args:
            chunk_ids: List of chunk document IDs
            
        Returns:
            List of RetrievalResult objects
        """
        pass

    def get_strategy_name(self) -> str:
        """Return the name of this retrieval strategy."""
        return self.__class__.__name__
