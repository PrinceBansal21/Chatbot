"""Indexing orchestration: chunking, embedding, vector-DB indexing, image chunks."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .chunking import ChunkingStrategyWrapper
from .embedding import EmbeddingClient
from .vectordb_indexer import VectorDBIndexer
from services.image import ImageChunkGenerator

from utils import (
    ImageConfigResolver,
    MetadataEnricher,
)
from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class IndexingOrchestrator:
    """
    Handles the indexing pipeline stages:
      chunk → embed → index (text and image chunks).

    Accepts pre-built service instances created by Pipeline.__init__.
    """

    def __init__(
        self,
        chunker: ChunkingStrategyWrapper,
        embedder: EmbeddingClient,
        indexer: VectorDBIndexer,
        *,
        image_config_resolver: Optional[ImageConfigResolver] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
        self.image_config_resolver = image_config_resolver
        self.config = config or {}

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        file_id: str,
        markdown: str,
        metadata: Dict[str, str],
        chunking_strategy: str,
        file_type_config: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Chunk the markdown document.

        Args:
            file_id: Document identifier
            markdown: Markdown content to chunk
            metadata: Document metadata
            chunking_strategy: Strategy name to use
            file_type_config: File-type-specific configuration with chunking options
        """
        LOGGER.log_chunking(chunking_strategy)

        chunking_options = {}
        if file_type_config:
            chunking_config = file_type_config.get("chunking", {})
            for key, value in chunking_config.items():
                if key != "strategy":
                    chunking_options[key] = value

        chunks = self.chunker.chunk(
            file_id=file_id,
            markdown=markdown,
            base_metadata=metadata,
            strategy=chunking_strategy,
            **chunking_options,
        )
        LOGGER.log_chunking_complete(len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list, embedding_provider: str, *, vector_dimension: int | None = None) -> list:
        """Generate embeddings for chunks."""
        LOGGER.log_embedding(embedding_provider, len(chunks))
        embeddings = self.embedder.embed(chunks, provider=embedding_provider, vector_dimension=vector_dimension)
        LOGGER.log_embedding_complete(len(embeddings))
        return embeddings

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_embeddings(self, embeddings: list, target_index: str) -> None:
        """Index embeddings in vector database."""
        LOGGER.log_indexing(target_index, len(embeddings))
        self.indexer.upsert(embeddings, index_name=target_index)
        LOGGER.log_indexing_complete(target_index)

    # ------------------------------------------------------------------
    # Image chunks
    # ------------------------------------------------------------------

    def create_and_index_image_chunks(
        self,
        file_id: str,
        image_manifest: Dict[str, Any],
        metadata: Dict[str, str],
        strategies: Dict[str, str],
        target_indices: List[str],
    ) -> int:
        """
        Create image chunks, embed ONCE, and index to ALL target indices.

        Returns:
            Number of image chunks created
        """
        doc_summary = metadata.get("document_summary")

        chunk_fields_cfg = self.config.get("chunk_fields", {})
        image_fields = MetadataEnricher.extract_field_names(
            chunk_fields_cfg.get("image_chunk_fields", {})
        )

        image_chunks = ImageChunkGenerator.create_image_chunks(
            file_id=file_id,
            image_manifest=image_manifest,
            base_metadata=metadata,
            page_summary=doc_summary,
            configured_fields=image_fields if image_fields else None,
        )

        if not image_chunks:
            LOGGER.log_stage(ProcessLogger.STAGE_CHUNK, "No image chunks created")
            return 0

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Created {len(image_chunks)} dedicated image chunks",
        )

        # Embed ONCE
        image_embeddings = self.embed_chunks(
            image_chunks, strategies["embedding_provider"],
            vector_dimension=strategies.get("vector_dimension"),
        )

        # Index to ALL target indices
        for target_index in target_indices:
            self.index_embeddings(image_embeddings, target_index)
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Indexed {len(image_chunks)} image chunks to {target_index}",
            )

        return len(image_chunks)

    # ------------------------------------------------------------------
    # Infographic page chunks
    # ------------------------------------------------------------------

    def create_and_index_infographic_chunks(
        self,
        file_id: str,
        file_path: str,
        infographic_pages: list,
        image_repository,
        metadata: dict,
        strategies: dict,
        target_indices: list,
        *,
        region: str = "us-east-1",
        vision_model: str | None = None,
    ) -> int:
        """Create, embed and index infographic page chunks.

        This calls :meth:`ImageChunkGenerator.create_infographic_page_chunks`
        which produces *two* chunks per page — one text-embedded and one
        image-embedded.  The :meth:`embed_chunks` call automatically routes
        them through the correct embedding path (the Cohere adapter detects
        ``image_base64`` on each chunk).

        Args:
            file_id: Document identifier.
            file_path: Local path to the PDF.
            infographic_pages: List of 1-based page numbers.
            image_repository: :class:`ImageRepository` for object storage.
            metadata: Base metadata dict.
            strategies: Dict with ``embedding_provider`` key.
            target_indices: Vector database indices to write to.
            region: Optional cloud provider region for the Vision LLM.
            vision_model: LLM model ID override. When ``None`` the
                callee resolves both model_id and region from config.

        Returns:
            Number of infographic chunks created.
        """
        if not infographic_pages:
            return 0

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Creating infographic chunks for pages {infographic_pages}",
        )

        # If the caller didn't supply an explicit vision_model, pass region=None
        # too so the callee resolves both strictly from model_defaults config
        # (no silent us-east-1 default).
        infographic_chunks = ImageChunkGenerator.create_infographic_page_chunks(
            file_id=file_id,
            file_path=file_path,
            page_numbers=infographic_pages,
            image_repository=image_repository,
            base_metadata=metadata,
            region=region if vision_model is not None else None,
            vision_model=vision_model,
        )

        if not infographic_chunks:
            LOGGER.log_stage(ProcessLogger.STAGE_CHUNK, "No infographic chunks created")
            return 0

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Created {len(infographic_chunks)} infographic chunks "
            f"({len(infographic_pages)} pages)",
        )

        # Embed (adapter automatically routes text vs image chunks)
        infographic_embeddings = self.embed_chunks(
            infographic_chunks, strategies["embedding_provider"],
            vector_dimension=strategies.get("vector_dimension"),
        )

        # Index to ALL target indices
        for target_index in target_indices:
            self.index_embeddings(infographic_embeddings, target_index)
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Indexed {len(infographic_chunks)} infographic chunks to {target_index}",
            )

        return len(infographic_chunks)

    # ------------------------------------------------------------------
    # Web / HTML infographic chunks
    # ------------------------------------------------------------------

    def create_and_index_web_infographic_chunks(
        self,
        file_id: str,
        image_manifest: dict,
        metadata: dict,
        strategies: dict,
        target_indices: list,
    ) -> int:
        """Create, embed and index infographic chunks for web/HTML images.

        Delegates to :meth:`ImageChunkGenerator.create_web_infographic_chunks`
        which produces up to two chunks per infographic image (text + image
        embedded).

        Args:
            file_id: Document identifier.
            image_manifest: Image manifest dict (with ``"images"`` list).
            metadata: Base metadata dict.
            strategies: Dict with ``embedding_provider`` key.
            target_indices: Vector database indices to write to.

        Returns:
            Number of web infographic chunks created.
        """
        if not image_manifest:
            return 0

        infographic_chunks = ImageChunkGenerator.create_web_infographic_chunks(
            file_id=file_id,
            image_manifest=image_manifest,
            base_metadata=metadata,
        )

        if not infographic_chunks:
            LOGGER.log_stage(ProcessLogger.STAGE_CHUNK, "No web infographic chunks created")
            return 0

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Created {len(infographic_chunks)} web infographic chunks",
        )

        # Embed (adapter automatically routes text vs image chunks)
        infographic_embeddings = self.embed_chunks(
            infographic_chunks, strategies["embedding_provider"],
            vector_dimension=strategies.get("vector_dimension"),
        )

        # Index to ALL target indices
        for target_index in target_indices:
            self.index_embeddings(infographic_embeddings, target_index)
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Indexed {len(infographic_chunks)} web infographic chunks to {target_index}",
            )

        return len(infographic_chunks)

    # ------------------------------------------------------------------
    # Image relevance tagging
    # ------------------------------------------------------------------

    def tag_image_relevance(
        self,
        image_manifest: dict,
        chunks: list,
        *,
        region: str | None = None,
        vision_model: str | None = None,
    ) -> dict:
        """Tag images in ``image_manifest`` as related or unrelated.

        Delegates to :meth:`ImageChunkGenerator.tag_image_relevance`.
        The manifest is modified **in place** — each image dict gets
        ``relevance`` and ``relevance_reason`` keys.

        Args:
            image_manifest: Full image manifest dict (with ``"images"`` key).
            chunks: Text chunks from the same document (for context).
            region: Optional cloud provider region.
            vision_model: LLM model ID override. When ``None``, cloud_lib
                resolves the active provider's configured vision model.

        Returns:
            The (modified) image manifest.
        """
        images = image_manifest.get("images", [])
        if not images:
            return image_manifest

        ImageChunkGenerator.tag_image_relevance(
            images=images,
            chunks=chunks,
            region=region if vision_model is not None else None,
            vision_model=vision_model,
        )

        return image_manifest
