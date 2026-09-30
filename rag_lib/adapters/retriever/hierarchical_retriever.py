"""
Hierarchical retriever with parent-child chunk expansion.

Designed for hierarchical chunking strategy that produces:
- Summary chunks (document-level)
- Parent chunks (section-level, larger context)
- Child chunks (fine-grained, for precise matching)

Retrieval strategy:
1. Search on child chunks for precise matching
2. Expand to parent chunks for fuller context
3. Optionally include document summaries
"""

from typing import List, Dict, Any, Optional, Callable, Set
from rag_lib.interfaces.retriever_manager import RetrievalQuery, RetrievalResult
from rag_lib.adapters.retriever.base_retriever import BaseRetriever


class HierarchicalRetriever(BaseRetriever):
    """
    Hierarchical retriever with parent-child expansion.

    Search Strategy:
    - Primary search on child chunks (fine-grained matching)
    - Expand results by fetching parent chunks
    - Deduplicate by parent to avoid redundancy
    - Optionally include document summaries
    """

    def __init__(
        self,
        vectordb_client: Any,
        index_name: str,
        embed_query_fn: Callable[[str], List[float]],
        vector_field: str = "vector",
        content_field: str = "content",
        chunk_type_field: str = "metadata.chunk_type",
        parent_id_field: str = "metadata.parent_id",
        include_summaries: bool = False,
        expand_to_parent: bool = True,
        child_multiplier: int = 3,  # Fetch 3x children to get enough unique parents
    ):
        """
        Initialize hierarchical retriever.

        Args:
            vectordb_client: VectorDB manager instance
            index_name: Name of the index to search
            embed_query_fn: Function to convert query text to embedding
            vector_field: Name of the vector field in the index
            content_field: Name of the content field in the index
            chunk_type_field: Field path for chunk type (summary/parent/child)
            parent_id_field: Field path for parent chunk ID reference
            include_summaries: Whether to include document summaries in results
            expand_to_parent: Whether to expand child results to parent chunks
            child_multiplier: Multiplier for initial child search (to ensure enough unique parents)
        """
        super().__init__(
            vectordb_client=vectordb_client,
            index_name=index_name,
            embed_query_fn=embed_query_fn,
            vector_field=vector_field,
            content_field=content_field,
        )
        self.chunk_type_field = chunk_type_field
        self.parent_id_field = parent_id_field
        self.include_summaries = include_summaries
        self.expand_to_parent = expand_to_parent
        self.child_multiplier = child_multiplier

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve chunks using hierarchical expansion.

        Strategy:
        1. Search child chunks for precise matching
        2. Group by parent and take top-scoring child per parent
        3. Fetch parent chunks for expanded context
        4. Optionally prepend document summaries

        Args:
            query: RetrievalQuery with query text and parameters

        Returns:
            List of RetrievalResult with parent context attached
        """
        query_vector = self._get_query_vector(query)

        # Step 1: Search child chunks (fetch more to ensure variety after parent dedup)
        child_k = query.top_k * self.child_multiplier

        child_hits = self._search_with_filter(
            query_vector=query_vector,
            k=child_k,
            chunk_type="child",
            extra_filters=query.filters,
        )

        child_results = self._hits_to_results(child_hits)

        if not self.expand_to_parent or not query.include_parent:
            # Return child results directly without expansion
            return child_results[: query.top_k]

        # Step 2: Group by parent and deduplicate
        parent_to_best_child: Dict[str, RetrievalResult] = {}
        children_without_parent: List[RetrievalResult] = []

        for result in child_results:
            parent_id = result.metadata.get("parent_id")

            if parent_id:
                # Keep highest scoring child per parent
                if parent_id not in parent_to_best_child:
                    parent_to_best_child[parent_id] = result
                elif result.score > parent_to_best_child[parent_id].score:
                    parent_to_best_child[parent_id] = result
            else:
                children_without_parent.append(result)

        # Step 3: Fetch parent chunks
        parent_ids = list(parent_to_best_child.keys())[: query.top_k]
        parent_results = self.retrieve_by_ids(parent_ids) if parent_ids else []

        # Create parent lookup
        parent_lookup = {r.chunk_id: r for r in parent_results}

        # Step 4: Build final results with parent content
        final_results: List[RetrievalResult] = []

        for parent_id in parent_ids:
            child_result = parent_to_best_child[parent_id]
            parent_result = parent_lookup.get(parent_id)

            # Attach parent content to child result
            final_result = RetrievalResult(
                content=child_result.content,
                score=child_result.score,
                metadata=child_result.metadata,
                chunk_id=child_result.chunk_id,
                parent_content=parent_result.content if parent_result else None,
            )
            final_results.append(final_result)

        # Add children without parents if we need more results
        remaining_slots = query.top_k - len(final_results)
        if remaining_slots > 0:
            final_results.extend(children_without_parent[:remaining_slots])

        # Step 5: Optionally prepend document summaries
        if self.include_summaries:
            summary_results = self._get_document_summaries(final_results, query_vector)
            # Summaries go first, then regular results
            final_results = summary_results + final_results

        return final_results[: query.top_k]

    def _search_with_filter(
        self,
        query_vector: List[float],
        k: int,
        chunk_type: str,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search with chunk_type filter."""
        filters = {"chunk_type": chunk_type}
        if extra_filters:
            filters.update(
                {
                    field: value
                    for field, value in extra_filters.items()
                    if field != "chunk_type"
                }
            )

        return self.vectordb.search_by_vector(
            self.index_name,
            query_vector,
            k=k,
            vector_field=self.vector_field,
            filters=filters,
        )

    def _get_document_summaries(
        self, results: List[RetrievalResult], query_vector: List[float]
    ) -> List[RetrievalResult]:
        """
        Get document summaries for the documents referenced in results.

        Returns unique summaries for documents appearing in results.
        """
        # Extract unique document/file identifiers
        doc_ids: Set[str] = set()
        for result in results:
            doc_id = result.metadata.get("file_id") or result.metadata.get(
                "document_id"
            )
            if doc_id:
                doc_ids.add(doc_id)

        if not doc_ids:
            return []

        # Search for summaries of these documents
        summary_results = []
        for doc_id in list(doc_ids)[:3]:  # Limit to 3 document summaries
            summary_hits = self._search_with_filter(
                query_vector=query_vector,
                k=1,
                chunk_type="summary",
                extra_filters={"file_id": doc_id},
            )
            if summary_hits:
                summary_results.extend(self._hits_to_results(summary_hits))

        return summary_results

    def retrieve_parents_only(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Alternative retrieval: search parent chunks directly.

        Useful when you want section-level matching rather than fine-grained.
        """
        query_vector = self._get_query_vector(query)

        parent_hits = self._search_with_filter(
            query_vector=query_vector,
            k=query.top_k,
            chunk_type="parent",
            extra_filters=query.filters,
        )

        return self._hits_to_results(parent_hits)

    def get_strategy_name(self) -> str:
        return "hierarchical"
