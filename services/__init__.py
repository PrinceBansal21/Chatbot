"""Service clients that orchestrate core business logic for document processing.

Organised into themed sub-packages:
- **config/**   – Configuration loading, file-type resolution, strategy resolution
- **document/** – Document fetching, Markdown conversion & storage, processing orchestration
- **image/**    – Image extraction, embedding in Markdown, chunk generation, cloud storage
- **indexing/** – Chunking, embedding generation, vector-DB indexing, indexing orchestration
- **llm/**      – LLM Converse client, summarisation, vision
- **storage/**  – Database persistence for ingestion tracking

Top-level standalone modules:
- ``graph_service``     – Knowledge-graph entity extraction and storage

All public symbols are re-exported here so existing ``from services import X``
statements continue to work unchanged.
"""

# -- config ------------------------------------------------------------------
from .config import ConfigLoader, FileTypeConfigResolver, StrategyResolver

# -- document ----------------------------------------------------------------
from .document import DocumentFetcher, DocumentProcessor, FetchedDocumentBundle, MarkdownConverter, MarkdownRepository

# -- image -------------------------------------------------------------------
from .image import ImageChunkGenerator, ImageProcessor, ImageRepository, MarkdownImageEmbedder

# -- indexing ----------------------------------------------------------------
from .indexing import Chunk, ChunkingStrategyWrapper, EmbeddingClient, IndexingOrchestrator, VectorDBIndexer

# -- llm ---------------------------------------------------------------------
from .llm import LLMConverseClient, ImageDescriptionAugmenter, InfographicClassifier, SummaryGenerator

# -- storage -----------------------------------------------------------------
from .storage import IngestionRepository

# -- top-level standalone ----------------------------------------------------
from .graph_service import GraphKnowledgeBaseService

__all__ = [
    # config
    "ConfigLoader",
    "FileTypeConfigResolver",
    "StrategyResolver",
    # document
    "DocumentFetcher",
    "DocumentProcessor",
    "FetchedDocumentBundle",
    "MarkdownConverter",
    "MarkdownRepository",
    # image
    "ImageChunkGenerator",
    "ImageProcessor",
    "ImageRepository",
    "MarkdownImageEmbedder",
    # indexing
    "Chunk",
    "ChunkingStrategyWrapper",
    "EmbeddingClient",
    "IndexingOrchestrator",
    "VectorDBIndexer",
    # llm
    "LLMConverseClient",
    "ImageDescriptionAugmenter",
    "InfographicClassifier",
    "SummaryGenerator",
    # storage
    "IngestionRepository",
    # top-level
    "GraphKnowledgeBaseService",
]
