"""Chunking service for Markdown content."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_lib.factory import RAGFactory
from rag_lib.interfaces.chunking_manager import Chunk
from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class ChunkingStrategyWrapper:
    """Wrapper to handle dynamic chunking strategy selection."""
    
    def __init__(self, config: Dict, embedding_client: Any):
        """
        Initialize the chunking strategy wrapper.
        
        Args:
            config: Chunking configuration dictionary
            embedding_client: Embedding client for strategies that need it (e.g., LLM chunking)
        """
        self.config = config
        self.default_strategy = (
            self.config.get("strategy")
            or self.config.get("default_strategy")
            or "recursive"
        ).lower()
        self.embedding_client = embedding_client
        self._managers = {}

    def _get_manager(self, strategy: str, override_config: Optional[Dict] = None):
        """
        Get or create a chunking manager for the specified strategy.
        
        Args:
            strategy: Chunking strategy name
            override_config: Optional config overrides (chunk_size, overlap, etc.)
        """
        # Create a cache key that includes overrides to avoid caching conflicts
        cache_key = strategy
        if override_config:
            # Include override values in cache key
            override_str = "_".join(f"{k}={v}" for k, v in sorted(override_config.items()) if v is not None)
            if override_str:
                cache_key = f"{strategy}_{override_str}"
        
        if cache_key not in self._managers:
            strategy_cfg = self.config.get(strategy, {})
            
            # Map common parameter names to rag_lib expected names
            # chunk_size -> max_tokens, overlap -> overlap_tokens
            max_tokens = (
                override_config.get("max_tokens") if override_config else None
            ) or (
                override_config.get("chunk_size") if override_config else None
            ) or strategy_cfg.get("max_tokens", 650 if strategy == "semantic" else 1800)
            
            overlap_tokens = (
                override_config.get("overlap_tokens") if override_config else None
            ) or (
                override_config.get("overlap") if override_config else None
            ) or strategy_cfg.get("overlap_tokens", 120)
            
            min_tokens = (
                override_config.get("min_tokens") if override_config else None
            ) or strategy_cfg.get("min_tokens", 250)
            
            chunking_kwargs = {
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "overlap_tokens": overlap_tokens,
            }
            
            if strategy == 'llm':
                chunking_kwargs['embedding_client'] = self.embedding_client
            
            # Forward all remaining strategy-specific config params to the adapter.
            # This is critical for strategies like 'contextual' that need llm_config,
            # parallel_requests, request_delay, batch_delay, etc.
            for key, value in strategy_cfg.items():
                if key not in chunking_kwargs:
                    chunking_kwargs[key] = value
                
            self._managers[cache_key] = RAGFactory.get_chunking_manager(
                strategy=strategy,
                **chunking_kwargs
            )
        return self._managers[cache_key]

    def chunk(
        self, 
        file_id: str, 
        markdown: str, 
        base_metadata: Dict[str, str], 
        strategy: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk markdown content using the specified or default strategy.
        
        Args:
            file_id: Unique identifier for the file
            markdown: Markdown content to chunk
            base_metadata: Base metadata to attach to each chunk
            strategy: Chunking strategy to use (optional, uses default if not specified)
            **kwargs: Additional chunking options:
                - chunk_size / max_tokens: Maximum tokens per chunk
                - overlap / overlap_tokens: Overlap between chunks
                - min_tokens: Minimum tokens per chunk
            
        Returns:
            List of Chunk objects
        """
        strategy = (strategy or self.default_strategy).lower()
        
        # Pass override config to manager creation
        override_config = kwargs if kwargs else None
        manager = self._get_manager(strategy, override_config)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Chunking with strategy '{strategy}'",
            extra_data={
                "content_length": len(markdown),
                "override_config": override_config,
            },
        )
        
        chunks = manager.chunk(file_id, markdown, base_metadata)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Chunking complete: {len(chunks)} chunks created",
            extra_data={
                "strategy": strategy,
                "chunk_count": len(chunks),
                "avg_content_length": (
                    sum(len(c.content) for c in chunks) // len(chunks)
                    if chunks else 0
                ),
            },
        )
        
        return chunks


__all__ = ["Chunk", "ChunkingStrategyWrapper"]
