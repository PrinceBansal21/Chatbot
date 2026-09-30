"""Graph Knowledge Base Service.

Orchestrates entity extraction, resolution, and storage during document
ingestion.  Sits between the pipeline (``app.py``) and the low-level
graph adapters in ``rag_lib``.

Usage in the pipeline::

    graph_service = GraphKnowledgeBaseService(config, region)
    ...
    graph_service.build_graph(chunks, file_id, index_name, doc_summary)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from rag_lib.factory import RAGFactory
from rag_lib.interfaces.graph_manager import (
    Entity,
    EntityExtractor,
    EntityResolver,
    GraphExtractionResult,
    GraphStore,
)

from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class GraphKnowledgeBaseService:
    """High-level service that wires extraction → resolution → storage."""

    # Stage constant for structured logging
    STAGE_GRAPH = "GRAPH"

    def __init__(
        self,
        config: Dict[str, Any],
        region: str,
    ):
        """
        Args:
            config: Full application config dict (must contain ``graph_knowledge_base``).
            region: Cloud region for the LLM used in entity extraction.
        """
        self._config = config
        self._region = region
        self._graph_cfg = config.get("graph_knowledge_base", {})
        self._enabled = self._graph_cfg.get("enabled", False)

        self._extractor: Optional[EntityExtractor] = None
        self._resolver: Optional[EntityResolver] = None
        self._store: Optional[GraphStore] = None

        if self._enabled:
            self._init_components()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Lazily initialise extractor, resolver, and store."""
        try:
            self._init_extractor()
            self._init_resolver()
            self._init_store()
            LOGGER.log_stage(
                ProcessLogger.STAGE_INIT,
                "Graph Knowledge Base Service initialised "
                f"(store={self._graph_cfg.get('store', {}).get('provider', 'in_memory')})",
            )
        except Exception as e:
            LOGGER.log_error(
                f"Failed to initialise Graph Knowledge Base Service: {e}",
                exception=e,
            )
            self._enabled = False

    def _init_extractor(self) -> None:
        ext_cfg = self._graph_cfg.get("extraction", {})
        llm_cfg = ext_cfg.get("llm", {})

        provider = llm_cfg.get("provider", "cloud")
        if provider != "cloud":
            raise ValueError(
                f"Unsupported graph extraction LLM provider {provider!r}; expected 'cloud'."
            )

        llm_manager = RAGFactory.get_llm_manager(
            provider=provider,
            model_key="graph_extraction",
            max_tokens=llm_cfg.get("max_tokens", 4096),
            temperature=llm_cfg.get("temperature", 0.0),
        )

        from rag_lib.adapters.entity_extraction import LLMEntityExtractor

        self._extractor = LLMEntityExtractor(
            llm_manager=llm_manager,
            max_tokens=ext_cfg.get("max_tokens", 4096),
            temperature=ext_cfg.get("temperature", 0.0),
            min_chunk_length=ext_cfg.get("min_chunk_length", 30),
            max_workers=ext_cfg.get("max_workers", 4),
        )

    def _init_resolver(self) -> None:
        res_cfg = self._graph_cfg.get("resolution", {})

        from rag_lib.adapters.entity_extraction.entity_resolver import (
            RuleBasedEntityResolver,
        )

        self._resolver = RuleBasedEntityResolver(
            similarity_threshold=res_cfg.get("similarity_threshold", 0.85),
            merge_aliases=res_cfg.get("merge_aliases", True),
        )

    def _init_store(self) -> None:
        store_cfg = self._graph_cfg.get("store", {})
        provider = store_cfg.get("provider", "in_memory")

        if provider == "gremlin":
            from rag_lib.adapters.graph import GremlinGraphAdapter

            self._store = GremlinGraphAdapter(
                endpoint=store_cfg.get("endpoint"),
                default_label=store_cfg.get("default_label", "kg"),
            )
        else:
            # Default: in-memory (for development / testing)
            from rag_lib.adapters.graph import InMemoryGraphStore

            self._store = InMemoryGraphStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def build_graph(
        self,
        chunks: List[Any],
        file_id: str,
        index_name: Optional[str] = None,
        document_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full extraction → resolution → storage pipeline.

        Args:
            chunks: List of ``Chunk`` objects (from the chunking step).
            file_id: Source document identifier.
            index_name: Graph namespace (defaults to the config default).
            document_summary: Optional document-level summary for context.

        Returns:
            Dict with keys:
            - ``entities_count``: number of unique entities stored
            - ``relationships_count``: number of relationships stored
            - ``extraction_time_s``: seconds spent on LLM extraction
            - ``total_time_s``: total wall-clock time
        """
        if not self._enabled:
            return {"skipped": True, "reason": "graph_knowledge_base disabled"}

        start = time.time()
        ext_cfg = self._graph_cfg.get("extraction", {})
        batch_size = ext_cfg.get("batch_size", 5)

        LOGGER.log_stage(
            self.STAGE_GRAPH,
            f"Starting graph extraction for {len(chunks)} chunks "
            f"(batch_size={batch_size})",
        )

        # 1. Extract
        extract_start = time.time()
        raw_extraction = self._extractor.extract(
            chunks,
            batch_size=batch_size,
            document_context=document_summary,
        )
        extraction_time = time.time() - extract_start

        LOGGER.log_stage(
            self.STAGE_GRAPH,
            f"Extracted {len(raw_extraction.entities)} entities, "
            f"{len(raw_extraction.relationships)} relationships "
            f"in {extraction_time:.1f}s",
        )

        if not raw_extraction.entities and not raw_extraction.relationships:
            LOGGER.log_stage(self.STAGE_GRAPH, "No entities/relationships found — skipping graph storage")
            return {
                "entities_count": 0,
                "relationships_count": 0,
                "extraction_time_s": round(extraction_time, 2),
                "total_time_s": round(time.time() - start, 2),
            }

        # 2. Resolve / deduplicate
        resolved = self._resolver.resolve(raw_extraction)

        LOGGER.log_stage(
            self.STAGE_GRAPH,
            f"Resolved to {len(resolved.entities)} unique entities, "
            f"{len(resolved.relationships)} unique relationships",
        )

        # 3. Delete existing graph data for this file (idempotent re-ingestion)
        deleted = self._store.delete_by_file_id(file_id, index_name=index_name)
        if deleted:
            LOGGER.log_stage(
                self.STAGE_GRAPH,
                f"Deleted {deleted} existing graph elements for file_id={file_id}",
            )

        # 4. Store
        counts = self._store.upsert(resolved, file_id=file_id, index_name=index_name)

        total_time = time.time() - start
        LOGGER.log_stage(
            self.STAGE_GRAPH,
            f"Graph build complete in {total_time:.1f}s: "
            f"{counts['entities_upserted']} entities, "
            f"{counts['relationships_upserted']} relationships",
        )

        return {
            "entities_count": counts["entities_upserted"],
            "relationships_count": counts["relationships_upserted"],
            "extraction_time_s": round(extraction_time, 2),
            "total_time_s": round(total_time, 2),
        }

    def delete_graph(
        self,
        file_id: str,
        index_name: Optional[str] = None,
    ) -> int:
        """Delete all graph data for a given file.

        Called during document deletion or re-ingestion.
        """
        if not self._enabled or not self._store:
            return 0
        return self._store.delete_by_file_id(file_id, index_name=index_name)

    def query_graph(
        self,
        entity_names: List[str],
        *,
        max_hops: int = 2,
        top_k: int = 20,
        index_name: Optional[str] = None,
    ):
        """Query the graph store.  Exposed for the chat retriever."""
        if not self._enabled or not self._store:
            from rag_lib.interfaces.graph_manager import GraphQueryResult
            return GraphQueryResult()
        return self._store.query(
            entity_names,
            max_hops=max_hops,
            top_k=top_k,
            index_name=index_name,
        )

    def close(self):
        """Release resources."""
        if self._store and hasattr(self._store, "close"):
            self._store.close()


__all__ = ["GraphKnowledgeBaseService"]
