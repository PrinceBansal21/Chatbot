from abc import ABC, abstractmethod
from typing import List, Tuple
from rag_lib.interfaces.chunking_manager import Chunk

class EmbeddingsManager(ABC):
    """Interface for embedding generation strategies."""

    @abstractmethod
    def embed(self, chunks: List[Chunk]) -> List[Tuple[Chunk, List[float]]]:
        """Generates embeddings for a list of chunks."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single query string."""
        pass
