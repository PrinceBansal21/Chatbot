"""
Base retriever with common functionality for all retriever implementations.
"""

from typing import List, Dict, Any, Optional, Callable
from rag_lib.interfaces.retriever_manager import (
    RetrieverManager,
    RetrievalQuery,
    RetrievalResult,
)


class BaseRetriever(RetrieverManager):
    """
    Base class providing common retriever functionality.

    Subclasses must implement the actual search logic.
    This class provides:
    - VectorDB client management
    - Embedding function integration
    - Common utility methods
    """

    def __init__(
        self,
        vectordb_client: Any,
        index_name: str,
        embed_query_fn: Callable[[str], List[float]],
        vector_field: str = "vector",
        content_field: str = "content",
    ):
        """
        Initialize base retriever.

        Args:
            vectordb_client: VectorDB manager instance (a cloud_lib VectorDBManager)
            index_name: Name of the index to search
            embed_query_fn: Function to convert query text to embedding vector
            vector_field: Name of the vector field in the index
            content_field: Name of the content/text field in the index
        """
        self.vectordb = vectordb_client
        self.index_name = index_name
        self.embed_query = embed_query_fn
        self.vector_field = vector_field
        self.content_field = content_field

    def _get_query_vector(self, query: RetrievalQuery) -> List[float]:
        """Get or compute query vector."""
        if query.query_vector is not None:
            return query.query_vector
        return self.embed_query(query.query_text)

    def _build_knn_query(
        self,
        query_vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a vector-database k-NN query body with proper sorting.

        CRITICAL FIX: Adds "sort" parameter to ensure chunks are returned
        in document sequence (by 'order' field) rather than relevance rank.
        This prevents RAG validation failures due to jumbled chunk order.

        NOTE: The 'order' field is top-level in the document, NOT nested
        under metadata. This is critical for the sort to work correctly.
        """
        knn_query = {
            "size": k,
            "query": {"knn": {self.vector_field: {"vector": query_vector, "k": k}}},
            # ✓ CRITICAL: Sort by 'order' field (top-level) to preserve document sequence
            "sort": [
                {"order": {"order": "asc"}},  # Primary: document order
                {"_score": {"order": "desc"}},  # Tiebreaker: relevance score
            ],
        }

        # Add filters if provided
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {f"metadata.{field}": value}})
                else:
                    filter_clauses.append({"term": {f"metadata.{field}": value}})

            if filter_clauses:
                knn_query["query"] = {
                    "bool": {"must": [knn_query["query"]], "filter": filter_clauses}
                }

        return knn_query

    def _hits_to_results(self, hits: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """Convert raw search hits to RetrievalResult objects."""
        results = []
        for hit in hits:
            content = hit.get(self.content_field, "")
            metadata = hit.get("metadata", {})
            chunk_id = hit.get("_id")
            score = hit.get("_score", 0.0)

            # Preserve top-level fields in metadata for neighbor expansion
            if "order" not in metadata and "order" in hit:
                metadata["_order"] = hit["order"]
            if "file_id" not in metadata and "file_id" in hit:
                metadata["_file_id"] = hit["file_id"]

            results.append(
                RetrievalResult(
                    content=content,
                    score=score,
                    metadata=metadata,
                    chunk_id=chunk_id,
                )
            )

        return results

    def _search_raw(
        self,
        query_vector: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
        custom_body: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute raw search and return hits with scores preserved.

        Calls the underlying vector-database client directly to retain _score,
        which is stripped by the cloud_lib adapter.

        Args:
            query_vector: The query embedding vector
            k: Number of results
            filters: Optional metadata filters
            custom_body: Optional custom query body (overrides default)

        Returns:
            List of raw hit dictionaries (with _score included)
        """
        if custom_body:
            query_body = custom_body
        else:
            query_body = self._build_knn_query(query_vector, k, filters)

        # search_documents() returns the raw response dict (hits include _score)
        response = self.vectordb.search_documents(self.index_name, query_body, size=k)
        results = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            source["_id"] = hit["_id"]
            source["_score"] = hit.get("_score", 0.0)
            results.append(source)
        return results

    def expand_with_neighbors(
        self,
        results: List[RetrievalResult],
        window_size: int = 1,
    ) -> List[RetrievalResult]:
        """
        Expand each retrieved chunk with neighboring chunks from the same document.

        For each result, fetches ``window_size`` chunks before and after
        based on the ``order`` field, then prepends/appends their content
        so the LLM sees the full surrounding context.

        Args:
            results: Retrieved chunks to expand.
            window_size: Number of neighbor chunks on each side (default 1).

        Returns:
            List of RetrievalResult with expanded content and
            ``metadata['neighbor_context']`` containing the merged text.
        """
        if not results or window_size <= 0:
            return results

        import logging

        logger = logging.getLogger(__name__)

        expanded: List[RetrievalResult] = []
        # Cache neighbor queries per (file_id, order) to avoid duplicate lookups
        _neighbor_cache: Dict[str, List[Dict[str, Any]]] = {}

        for result in results:
            meta = result.metadata or {}
            file_id = (
                meta.get("file_id")
                or meta.get("_file_id")
                or meta.get("original_file_id")
            )
            order = meta.get("_order")

            # If order is missing, try to parse from chunk_id pattern "{file_id}-{order:05d}"
            if order is None and result.chunk_id and file_id:
                suffix = result.chunk_id.replace(file_id + "-", "", 1)
                # Only take the first segment (handles extra suffixes)
                parts = suffix.split("-")
                if parts and parts[0].isdigit():
                    order = int(parts[0])

            if file_id is None or order is None:
                # Cannot determine neighbors — return as-is
                expanded.append(result)
                continue

            cache_key = f"{file_id}:{order}:{window_size}"
            if cache_key in _neighbor_cache:
                neighbor_hits = _neighbor_cache[cache_key]
            else:
                try:
                    neighbor_hits = self.vectordb.get_neighboring_documents(
                        self.index_name,
                        document_field="file_id",
                        document_id=file_id,
                        order_field="order",
                        order=order,
                        window_size=window_size,
                        source_fields=[
                            self.content_field,
                            "order",
                            "file_id",
                            "metadata",
                        ],
                    )
                except Exception as e:
                    logger.warning(f"Neighbor chunk expansion failed: {e}")
                    neighbor_hits = []

                _neighbor_cache[cache_key] = neighbor_hits

            # Build expanded content (before → current → after)
            before_parts = []
            after_parts = []
            for nb in neighbor_hits:
                nb_order = nb.get("order")
                nb_content = nb.get(self.content_field, "")
                if nb_order is not None and nb_content:
                    if nb_order < order:
                        before_parts.append((nb_order, nb_content))
                    elif nb_order > order:
                        after_parts.append((nb_order, nb_content))

            before_parts.sort(key=lambda x: x[0])
            after_parts.sort(key=lambda x: x[0])

            before_text = "\n\n".join(t for _, t in before_parts)
            after_text = "\n\n".join(t for _, t in after_parts)

            # Merge into a single expanded context
            parts = []
            if before_text:
                parts.append(before_text)
            parts.append(result.content)
            if after_text:
                parts.append(after_text)
            merged_content = "\n\n".join(parts)

            new_meta = dict(meta)
            new_meta["neighbor_context"] = True
            new_meta["neighbor_window"] = window_size
            new_meta["original_content"] = result.content

            expanded.append(
                RetrievalResult(
                    content=merged_content,
                    score=result.score,
                    metadata=new_meta,
                    chunk_id=result.chunk_id,
                    parent_content=result.parent_content,
                )
            )

            if len(before_parts) + len(after_parts) > 0:
                logger.debug(
                    f"Expanded chunk {result.chunk_id}: +{len(before_parts)} before, "
                    f"+{len(after_parts)} after"
                )

        return expanded

    def retrieve_by_ids(self, chunk_ids: List[str]) -> List[RetrievalResult]:
        """
        Retrieve specific chunks by their IDs.

        Uses the vector database's multi-get or terms query.
        """
        if not chunk_ids:
            return []

        try:
            hits = self.vectordb.get_documents_by_ids(self.index_name, chunk_ids)
            return self._hits_to_results(hits)
        except Exception as e:
            # Log error and return empty
            print(f"Error retrieving by IDs: {e}")
            return []
