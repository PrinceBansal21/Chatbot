"""Embedding service for generating vector embeddings."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from rag_lib.factory import RAGFactory
from rag_lib.interfaces.chunking_manager import Chunk

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


class EmbeddingClient:
    """Creates vector embeddings using RAGFactory."""

    def __init__(self, config: Dict):
        self.config = config.copy()
        self.default_provider = (
            self.config.get("provider")
            or self.config.get("default_provider")
            or "cloud"
        )
        self._managers = {}

    def _get_manager(self, provider: str, output_dimension: int | None = None):
        manager_key = (provider, output_dimension)
        if manager_key not in self._managers:
            # Extract provider-specific configuration
            provider_config = self.config.get(provider, {}).copy()
            
            # Map 'region' to 'region_name' for cloud_lib adapter compatibility
            if "region" in provider_config:
                provider_config["region_name"] = provider_config.pop("region")
            
            # Remove 'endpoint' if it is None, as Pydantic validation fails on extra fields
            if "endpoint" in provider_config and provider_config["endpoint"] is None:
                provider_config.pop("endpoint")
            if output_dimension is not None:
                provider_config["output_dimension"] = output_dimension

            self._managers[manager_key] = RAGFactory.get_embeddings_manager(
                provider, **provider_config
            )
        return self._managers[manager_key]

    def embed(self, chunks: Iterable[Chunk], provider: str = None, vector_dimension: int | None = None) -> List[Tuple[Chunk, List[float]]]:
        """
        Generates embeddings for the provided chunks.
        
        Args:
            chunks: Iterable of Chunk objects to embed.
            provider: Optional provider to use for embedding.
            vector_dimension: Optional output dimension for the embedding model.
            
        Returns:
            List of tuples containing (Chunk, embedding_vector).
        """
        provider = provider or self.default_provider
        manager = self._get_manager(provider, vector_dimension)
        
        chunk_list = list(chunks)
        if not chunk_list:
            LOGGER.log_stage(
                ProcessLogger.STAGE_EMBED,
                "No chunks to embed, returning empty list",
            )
            return []

        LOGGER.log_stage(
            ProcessLogger.STAGE_EMBED,
            f"Generating embeddings for {len(chunk_list)} chunks using provider '{provider}'",
        )
        
        import time
        start = time.time()
        results = manager.embed(chunk_list)
        elapsed = time.time() - start
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_EMBED,
            f"Embedding complete: {len(results)} vectors generated in {elapsed:.1f}s",
            extra_data={
                "provider": provider,
                "chunk_count": len(chunk_list),
                "vector_count": len(results),
                "duration_seconds": round(elapsed, 2),
            },
        )
        
        return results
