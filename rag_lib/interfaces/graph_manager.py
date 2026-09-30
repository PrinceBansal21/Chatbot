"""Graph Knowledge Base Manager Interface.

Defines the contracts for graph-based knowledge extraction, storage,
and retrieval. Supports entity/relationship extraction from text chunks
and graph traversal for retrieval-augmented generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from text.

    Attributes:
        name: Canonical entity name (e.g. "United Nations")
        entity_type: Category label (PERSON, ORG, METRIC, DATE, CONCEPT, PRODUCT, LOCATION, EVENT, etc.)
        aliases: Alternative surface forms found in text (e.g. ["UN", "the United Nations"])
        properties: Arbitrary key-value metadata (description, source_chunk_id, …)
    """
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A directed relationship between two entities.

    Attributes:
        subject: Source entity name
        predicate: Relationship label (e.g. "reported", "acquired", "is_part_of")
        object: Target entity name
        properties: Extra metadata (confidence, source_chunk_id, evidence text, …)
    """
    subject: str
    predicate: str
    object: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphExtractionResult:
    """Result of entity/relationship extraction from a batch of chunks.

    Attributes:
        entities: Extracted entities
        relationships: Extracted relationships
        source_chunk_ids: Chunk IDs this result was extracted from
        metadata: Additional extraction metadata (model_id, token_usage, …)
    """
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQueryResult:
    """Result from a graph knowledge-base query.

    Attributes:
        entities: Entities matching or related to the query
        relationships: Relationships connecting matched entities
        subgraph_text: Human-readable serialisation of the matched sub-graph
            (for injection into the LLM context window)
        source_chunk_ids: Chunk IDs associated with the matched triples
        metadata: Query execution metadata (hops, latency, …)
    """
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    subgraph_text: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class EntityExtractor(ABC):
    """Extracts entities and relationships from text chunks using an LLM."""

    @abstractmethod
    def extract(
        self,
        chunks: List[Any],
        *,
        batch_size: int = 5,
        document_context: Optional[str] = None,
    ) -> GraphExtractionResult:
        """Extract entities and relationships from a list of ``Chunk`` objects.

        Args:
            chunks: List of ``Chunk`` dataclass instances (from chunking adapters).
            batch_size: Number of chunks to send per LLM call for batching.
            document_context: Optional document-level context (title, summary)
                to improve extraction quality.

        Returns:
            ``GraphExtractionResult`` containing all extracted entities and relationships.
        """
        pass


class GraphStore(ABC):
    """Persists and queries a property-graph knowledge base."""

    @abstractmethod
    def upsert(
        self,
        extraction: GraphExtractionResult,
        *,
        file_id: str,
        index_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """Insert or update entities and relationships in the graph store.

        Args:
            extraction: The ``GraphExtractionResult`` to persist.
            file_id: Source document identifier (used for provenance tracking).
            index_name: Optional namespace / collection name.

        Returns:
            Dict with counts: ``{"entities_upserted": N, "relationships_upserted": M}``
        """
        pass

    @abstractmethod
    def query(
        self,
        query_entities: List[str],
        *,
        max_hops: int = 2,
        top_k: int = 20,
        index_name: Optional[str] = None,
    ) -> GraphQueryResult:
        """Traverse the graph from seed entities and return relevant sub-graph.

        Args:
            query_entities: List of entity names (or aliases) to start traversal from.
            max_hops: Maximum relationship hops from seed entities.
            top_k: Maximum number of relationships to return.
            index_name: Optional namespace / collection name.

        Returns:
            ``GraphQueryResult`` containing the matched sub-graph.
        """
        pass

    @abstractmethod
    def delete_by_file_id(
        self,
        file_id: str,
        *,
        index_name: Optional[str] = None,
    ) -> int:
        """Remove all entities and relationships originating from a document.

        Args:
            file_id: The document whose graph data should be deleted.
            index_name: Optional namespace / collection name.

        Returns:
            Number of graph elements deleted.
        """
        pass

    def get_entity(self, name: str) -> Optional[Entity]:
        """Look up a single entity by canonical name (optional override)."""
        return None


class EntityResolver(ABC):
    """Deduplicates and normalises entities across extraction batches."""

    @abstractmethod
    def resolve(
        self,
        extraction: GraphExtractionResult,
        *,
        existing_entities: Optional[List[Entity]] = None,
    ) -> GraphExtractionResult:
        """Deduplicate entities and remap relationships to canonical names.

        Args:
            extraction: Raw extraction result to resolve.
            existing_entities: Previously resolved entities (for cross-batch dedup).

        Returns:
            A new ``GraphExtractionResult`` with resolved (canonical) entity names.
        """
        pass


__all__ = [
    "Entity",
    "Relationship",
    "GraphExtractionResult",
    "GraphQueryResult",
    "EntityExtractor",
    "GraphStore",
    "EntityResolver",
]
