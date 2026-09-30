"""Graph-augmented retriever.

Combines **vector similarity search** (standard k-NN through a vector database) with
**graph traversal** (entity-based sub-graph lookup) and merges the results
using Reciprocal Rank Fusion (RRF).

Flow:
1. Extract key entities from the user query (lightweight LLM call).
2. Vector search → top-K chunks by embedding similarity.
3. Graph query  → related entities, relationships, and source chunk IDs.
4. Fetch graph-related chunks from the vector store.
5. Merge & deduplicate via RRF scoring.
6. Prepend the graph sub-graph summary as structured context.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from rag_lib.interfaces.retriever_manager import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverManager,
)
from rag_lib.adapters.retriever.base_retriever import BaseRetriever
from rag_lib.interfaces.graph_manager import GraphQueryResult, GraphStore
from rag_lib.interfaces.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity extraction prompt for queries
# ---------------------------------------------------------------------------
_ENTITY_EXTRACTION_SYSTEM = (
    "You are an entity extractor. Given a user question, return a JSON array "
    "of the key entity names mentioned. Return ONLY a JSON array of strings, "
    "no commentary."
)
_ENTITY_EXTRACTION_TEMPLATE = (
    'Extract key entities (people, organisations, products, metrics, concepts) '
    'from this question:\n\n"{query}"\n\nReturn a JSON array of entity name strings.'
)


class GraphAugmentedRetriever(BaseRetriever):
    """Hybrid vector + graph retriever.

    Args:
        vectordb_client: Cloud-neutral vector database adapter.
        index_name: Vector database index name.
        embed_query_fn: Query embedding function.
        graph_store: ``GraphStore`` instance for graph traversal.
        llm_manager: LLM for extracting entities from queries.
        vector_field / content_field: Vector database field names.
        graph_weight: Weight of graph results in RRF (0–1). Default 0.4.
        max_hops: Max traversal depth in the graph.  Default 2.
        graph_top_k: Max relationships from graph query.  Default 20.
        rrf_k: RRF constant (higher = less aggressive fusion). Default 60.
        graph_index_name: Namespace in the graph store (defaults to OS index).
    """

    def __init__(
        self,
        vectordb_client: Any,
        index_name: str,
        embed_query_fn: Callable[[str], List[float]],
        graph_store: GraphStore,
        llm_manager: Optional[LLMManager] = None,
        vector_field: str = "vector",
        content_field: str = "content",
        graph_weight: float = 0.4,
        max_hops: int = 2,
        graph_top_k: int = 20,
        rrf_k: int = 60,
        graph_index_name: Optional[str] = None,
    ):
        super().__init__(
            vectordb_client=vectordb_client,
            index_name=index_name,
            embed_query_fn=embed_query_fn,
            vector_field=vector_field,
            content_field=content_field,
        )
        self._graph_store = graph_store
        self._llm = llm_manager
        self._graph_weight = graph_weight
        self._vector_weight = 1.0 - graph_weight
        self._max_hops = max_hops
        self._graph_top_k = graph_top_k
        self._rrf_k = rrf_k
        self._graph_index_name = graph_index_name or index_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using vector search + graph traversal + RRF fusion."""

        # 1. Vector search
        query_vector = self._get_query_vector(query)
        vector_hits = self._search_raw(
            query_vector=query_vector,
            k=query.top_k * 2,  # over-fetch for RRF merging
            filters=query.filters,
        )
        vector_results = self._hits_to_results(vector_hits)
        logger.info("Vector search returned %d results", len(vector_results))

        # 2. Extract entities from query
        query_entities = self._extract_query_entities(query.query_text)
        logger.info("Extracted query entities: %s", query_entities)

        if not query_entities:
            # No entities found — fall back to pure vector search
            return vector_results[: query.top_k]

        # 3. Graph traversal
        graph_result = self._graph_store.query(
            query_entities,
            max_hops=self._max_hops,
            top_k=self._graph_top_k,
            index_name=self._graph_index_name,
        )

        # 4. Fetch graph-referenced chunks from vector store
        graph_chunk_ids = graph_result.source_chunk_ids
        graph_chunk_results: List[RetrievalResult] = []
        if graph_chunk_ids:
            graph_chunk_results = self.retrieve_by_ids(graph_chunk_ids)
            logger.info(
                "Graph traversal found %d related chunks", len(graph_chunk_results)
            )

        # 5. Merge via RRF
        merged = self._rrf_merge(vector_results, graph_chunk_results, query.top_k)

        # 6. Inject graph sub-graph summary into the top result's metadata
        if graph_result.subgraph_text and merged:
            merged[0].metadata["graph_context"] = graph_result.subgraph_text
            merged[0].metadata["graph_entities"] = json.dumps(
                [e.name for e in graph_result.entities]
            )

        return merged

    def get_strategy_name(self) -> str:
        return "graph"

    # ------------------------------------------------------------------
    # Entity extraction from query
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query_text: str) -> List[str]:
        """Use LLM (or simple heuristics) to extract entity names from query."""
        if self._llm:
            return self._extract_via_llm(query_text)
        return self._extract_via_heuristics(query_text)

    def _extract_via_llm(self, query_text: str) -> List[str]:
        """LLM-based entity extraction from user query."""
        try:
            prompt = _ENTITY_EXTRACTION_TEMPLATE.format(query=query_text)
            response = self._llm.generate(
                prompt=prompt,
                system_prompt=_ENTITY_EXTRACTION_SYSTEM,
                max_tokens=256,
                temperature=0.0,
            )
            text = response.text.strip()
            # Strip markdown fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            entities = json.loads(text)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except Exception:
            logger.warning(
                "LLM entity extraction failed for query, falling back to heuristics"
            )
        return self._extract_via_heuristics(query_text)

    @staticmethod
    def _extract_via_heuristics(query_text: str) -> List[str]:
        """Simple heuristic: extract capitalised phrases as entity candidates."""
        # Match title-case words and multi-word phrases
        patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Multi-word proper nouns
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b[A-Z][a-z]{2,}\b",  # Single proper nouns
        ]
        entities = []
        for pattern in patterns:
            entities.extend(re.findall(pattern, query_text))
        # Deduplicate preserving order
        seen = set()
        result = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                result.append(e)
        return result

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Score vector results
        for rank, r in enumerate(vector_results, 1):
            key = r.chunk_id or r.content[:80]
            scores[key] = scores.get(key, 0) + self._vector_weight / (self._rrf_k + rank)
            result_map[key] = r

        # Score graph results
        for rank, r in enumerate(graph_results, 1):
            key = r.chunk_id or r.content[:80]
            scores[key] = scores.get(key, 0) + self._graph_weight / (self._rrf_k + rank)
            if key not in result_map:
                result_map[key] = r

        # Sort by RRF score descending
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        merged: List[RetrievalResult] = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.score = scores[key]  # Replace with RRF score
            result.metadata["retrieval_source"] = "graph+vector"
            merged.append(result)

        return merged


__all__ = ["GraphAugmentedRetriever"]
