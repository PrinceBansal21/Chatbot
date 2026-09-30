from rag_lib.adapters.retriever.hierarchical_retriever import HierarchicalRetriever
from rag_lib.adapters.retriever.hybrid_retriever import HybridRetriever
from rag_lib.adapters.retriever.standard_retriever import StandardRetriever
from rag_lib.interfaces.retriever_manager import RetrievalQuery, RetrievalResult


class FakeVectorManager:
    def __init__(self):
        self.calls = []

    def search_by_vector(self, index_name, query_vector, **kwargs):
        self.calls.append(("vector", index_name, query_vector, kwargs))
        return [
            {
                "_id": "chunk-2",
                "_score": 0.9,
                "content": "current",
                "file_id": "file-1",
                "order": 2,
                "metadata": {"file_id": "file-1"},
            }
        ]

    def search_hybrid(self, index_name, query_text, query_vector, text_fields, **kwargs):
        self.calls.append(
            ("hybrid", index_name, query_text, query_vector, text_fields, kwargs)
        )
        return [{"_id": "hybrid-1", "_score": 0.8, "content": "hybrid"}]

    def search_text(self, index_name, query_text, fields, **kwargs):
        self.calls.append(("text", index_name, query_text, fields, kwargs))
        return [{"_id": "text-1", "_score": 0.7, "content": "text"}]

    def get_neighboring_documents(self, index_name, **kwargs):
        self.calls.append(("neighbors", index_name, kwargs))
        return [
            {"content": "before", "order": 1},
            {"content": "current", "order": 2},
            {"content": "after", "order": 3},
        ]

    def get_documents_by_ids(self, index_name, document_ids):
        self.calls.append(("ids", index_name, document_ids))
        return [
            {"_id": document_ids[0], "_score": 1.0, "content": "hydrated"}
        ]


def test_standard_search_and_expansion_use_semantic_manager_operations():
    manager = FakeVectorManager()
    retriever = StandardRetriever(manager, "chunks", lambda _: [0.1])

    results = retriever.retrieve(
        RetrievalQuery("question", top_k=4, filters={"team": "support"})
    )
    assert manager.calls[0] == (
        "vector",
        "chunks",
        [0.1],
        {
            "k": 4,
            "vector_field": "vector",
            "filters": {"team": "support"},
            "sort_by": [{"order": "asc"}, {"_score": "desc"}],
        },
    )

    expanded = retriever.expand_with_neighbors(results, window_size=1)
    assert expanded[0].content == "before\n\ncurrent\n\nafter"
    assert manager.calls[1][0] == "neighbors"

    hydrated = retriever.retrieve_by_ids(["parent-1"])
    assert hydrated[0].chunk_id == "parent-1"
    assert hydrated[0].content == "hydrated"
    assert manager.calls[2] == ("ids", "chunks", ["parent-1"])


def test_hybrid_search_passes_fields_weights_and_keyword_boosts():
    manager = FakeVectorManager()
    retriever = HybridRetriever(manager, "chunks", lambda _: [0.2])

    hybrid = retriever._hybrid_search(
        "question", [0.2], 6, 0.65, {"team": ["a", "b"]}
    )
    assert hybrid[0].chunk_id == "hybrid-1"
    assert manager.calls[0] == (
        "hybrid",
        "chunks",
        "question",
        [0.2],
        ["content", "metadata.keywords"],
        {
            "k": 6,
            "vector_field": "vector",
            "vector_weight": 0.65,
            "filters": {"team": ["a", "b"]},
        },
    )

    keyword = retriever._keyword_search("question", 6, {"team": "a"})
    assert keyword[0].chunk_id == "text-1"
    assert manager.calls[1] == (
        "text",
        "chunks",
        "question",
        {"content": 1.0, "metadata.keywords": 1.5},
        {"k": 6, "filters": {"team": "a"}},
    )


def test_hierarchical_search_protects_chunk_type_filter():
    manager = FakeVectorManager()
    retriever = HierarchicalRetriever(manager, "chunks", lambda _: [0.3])

    retriever._search_with_filter(
        [0.3],
        9,
        "child",
        {"chunk_type": "parent", "department": "finance"},
    )

    assert manager.calls[0] == (
        "vector",
        "chunks",
        [0.3],
        {
            "k": 9,
            "vector_field": "vector",
            "filters": {"chunk_type": "child", "department": "finance"},
        },
    )
