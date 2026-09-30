"""
Hybrid retriever combining vector similarity and keyword search.

Designed for enriched chunking strategy that produces:
- Chunks with keywords in metadata
- Chunks with summaries in metadata
- Optional: structured metadata for filtering

Retrieval strategy:
1. Vector similarity search (semantic matching)
2. Keyword/BM25 search (lexical matching)
3. Combine scores with configurable alpha weighting
"""

from typing import List, Dict, Any, Optional, Callable
import logging
from rag_lib.interfaces.retriever_manager import RetrievalQuery, RetrievalResult
from rag_lib.adapters.retriever.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector and keyword search.

    Combines semantic similarity (k-NN) with lexical matching (BM25/keyword)
    for improved retrieval precision, especially with enriched chunks.
    """

    def __init__(
        self,
        vectordb_client: Any,
        index_name: str,
        embed_query_fn: Callable[[str], List[float]],
        vector_field: str = "vector",
        content_field: str = "content",
        keywords_field: str = "metadata.keywords",
        default_alpha: float = 0.7,  # 70% vector, 30% keyword
        use_keyword_boost: bool = True,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vectordb_client: VectorDB manager instance
            index_name: Name of the index to search
            embed_query_fn: Function to convert query text to embedding
            vector_field: Name of the vector field in the index
            content_field: Name of the content field in the index
            keywords_field: Field path for stored keywords
            default_alpha: Default balance (1.0=all vector, 0.0=all keyword)
            use_keyword_boost: Whether to boost matches on keyword field
        """
        super().__init__(
            vectordb_client=vectordb_client,
            index_name=index_name,
            embed_query_fn=embed_query_fn,
            vector_field=vector_field,
            content_field=content_field,
        )
        self.keywords_field = keywords_field
        self.default_alpha = default_alpha
        self.use_keyword_boost = use_keyword_boost

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve chunks using hybrid vector + keyword search.

        Strategy:
        1. Execute vector similarity search
        2. Execute keyword/BM25 search on content and keywords
        3. Combine results using Reciprocal Rank Fusion (RRF) or alpha blending

        Args:
            query: RetrievalQuery with query text and parameters

        Returns:
            List of RetrievalResult sorted by combined score
        """
        query_vector = self._get_query_vector(query)
        alpha = (
            query.hybrid_alpha if hasattr(query, "hybrid_alpha") else self.default_alpha
        )

        # Fetch more results for combination
        fetch_k = query.top_k * 2

        # Strategy 1: Pure hybrid query using the vector database's capabilities.
        try:
            results = self._hybrid_search(
                query_text=query.query_text,
                query_vector=query_vector,
                k=fetch_k,
                alpha=alpha,
                filters=query.filters,
            )
            return results[: query.top_k]
        except Exception:
            # Fallback to manual combination if hybrid not supported
            pass

        # Strategy 2: Manual combination (fallback)
        vector_results = self._vector_search(query_vector, fetch_k, query.filters)
        keyword_results = self._keyword_search(query.query_text, fetch_k, query.filters)

        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results, keyword_results, alpha=alpha, k=60  # RRF constant
        )

        return combined[: query.top_k]

    def _hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        k: int,
        alpha: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute a native hybrid search query.

        Uses bool query with both knn and match clauses.
        """
        # Build hybrid query
        hits = self.vectordb.search_hybrid(
            self.index_name,
            query_text,
            query_vector,
            [self.content_field, self.keywords_field],
            k=k,
            vector_field=self.vector_field,
            vector_weight=alpha,
            filters=filters,
        )
        return self._hits_to_results(hits)

    def _vector_search(
        self,
        query_vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Execute pure vector similarity search."""
        hits = self.vectordb.search_by_vector(
            self.index_name,
            query_vector,
            k=k,
            vector_field=self.vector_field,
            filters=filters,
        )
        return self._hits_to_results(hits)

    def _keyword_search(
        self, query_text: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Execute keyword/BM25 search."""
        fields = {self.content_field: 1.0}
        if self.use_keyword_boost:
            fields[self.keywords_field] = 1.5

        try:
            hits = self.vectordb.search_text(
                self.index_name,
                query_text,
                fields,
                k=k,
                filters=filters,
            )
            return self._hits_to_results(hits)
        except Exception as e:
            logger.warning("Keyword search error: %s", e)
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        alpha: float = 0.7,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF Score = alpha * (1 / (k + vector_rank)) + (1-alpha) * (1 / (k + keyword_rank))

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector results (1-alpha for keyword)
            k: RRF constant (default 60)

        Returns:
            Combined and re-ranked results
        """
        # Create rank maps
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        keyword_ranks = {r.chunk_id: i + 1 for i, r in enumerate(keyword_results)}

        # Collect all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())

        # Create result lookup
        result_lookup = {}
        for r in vector_results + keyword_results:
            if r.chunk_id not in result_lookup:
                result_lookup[r.chunk_id] = r

        # Calculate RRF scores
        rrf_scores = []
        for chunk_id in all_chunk_ids:
            vector_rank = vector_ranks.get(chunk_id, len(vector_results) + 100)
            keyword_rank = keyword_ranks.get(chunk_id, len(keyword_results) + 100)

            # RRF formula with alpha weighting
            rrf_score = alpha * (1.0 / (k + vector_rank)) + (1 - alpha) * (
                1.0 / (k + keyword_rank)
            )

            rrf_scores.append((chunk_id, rrf_score))

        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # Build final results
        final_results = []
        for chunk_id, score in rrf_scores:
            result = result_lookup[chunk_id]
            # Update score to RRF score
            final_results.append(
                RetrievalResult(
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                    chunk_id=result.chunk_id,
                    parent_content=result.parent_content,
                )
            )

        return final_results

    def get_strategy_name(self) -> str:
        return "hybrid"
