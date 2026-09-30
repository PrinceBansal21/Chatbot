"""
Standard vector similarity retriever.

Simple k-NN search without any special logic - suitable for:
- Recursive chunking
- Contextual chunking (context already embedded in content)
- Any strategy where chunks are independent
"""

from typing import List, Dict, Any, Optional, Callable
from rag_lib.interfaces.retriever_manager import RetrievalQuery, RetrievalResult
from rag_lib.adapters.retriever.base_retriever import BaseRetriever


class StandardRetriever(BaseRetriever):
    """
    Standard vector similarity retriever using k-NN search.

    This is the default retriever suitable for most chunking strategies
    where chunks are independent units (recursive, contextual, fixed-size, etc.)
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
        Initialize standard retriever.

        Args:
            vectordb_client: VectorDB manager instance
            index_name: Name of the index to search
            embed_query_fn: Function to convert query text to embedding
            vector_field: Name of the vector field in the index
            content_field: Name of the content field in the index
        """
        super().__init__(
            vectordb_client=vectordb_client,
            index_name=index_name,
            embed_query_fn=embed_query_fn,
            vector_field=vector_field,
            content_field=content_field,
        )

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve chunks using standard k-NN vector similarity.

        Args:
            query: RetrievalQuery with query text and parameters

        Returns:
            List of RetrievalResult sorted by similarity (descending)
        """
        # Get query vector
        query_vector = self._get_query_vector(query)

        hits = self.vectordb.search_by_vector(
            self.index_name,
            query_vector,
            k=query.top_k,
            vector_field=self.vector_field,
            filters=query.filters,
            sort_by=[{"order": "asc"}, {"_score": "desc"}],
        )

        # Convert to results
        return self._hits_to_results(hits)

    def get_strategy_name(self) -> str:
        return "standard"
