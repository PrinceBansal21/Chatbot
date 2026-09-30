"""Graph storage adapters.

``GremlinGraphAdapter`` stores entities as vertices and relationships as
edges in any Gremlin-compatible graph database (managed or self-hosted
Cosmos DB Gremlin API, JanusGraph, etc.). ``InMemoryGraphStore`` is a
lightweight in-process store for development/testing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rag_lib.adapters.graph.gremlin_adapter import GremlinGraphAdapter
from rag_lib.interfaces.graph_manager import (
    Entity,
    GraphExtractionResult,
    GraphQueryResult,
    GraphStore,
    Relationship,
)


# ---------------------------------------------------------------------------
# In-Memory Graph Store (for development / testing)
# ---------------------------------------------------------------------------

class InMemoryGraphStore(GraphStore):
    """Simple in-memory graph store for prototyping and testing.

    Stores entities and relationships in Python dicts. No persistence.
    """

    def __init__(self):
        # name_lower → Entity
        self._entities: Dict[str, Entity] = {}
        # (subj_lower, pred, obj_lower) → Relationship
        self._relationships: Dict[tuple, Relationship] = {}
        # file_id → set of entity name_lowers
        self._file_entities: Dict[str, set] = {}
        # file_id → set of relationship keys
        self._file_rels: Dict[str, set] = {}

    def upsert(
        self,
        extraction: GraphExtractionResult,
        *,
        file_id: str,
        index_name: Optional[str] = None,
    ) -> Dict[str, int]:
        ent_count = 0
        for entity in extraction.entities:
            key = entity.name.lower()
            if key in self._entities:
                existing = self._entities[key]
                merged_aliases = set(existing.aliases)
                merged_aliases.update(entity.aliases)
                merged_aliases.discard(existing.name)
                existing.aliases = sorted(merged_aliases)
            else:
                self._entities[key] = entity
                ent_count += 1
            self._file_entities.setdefault(file_id, set()).add(key)

        rel_count = 0
        for rel in extraction.relationships:
            key = (rel.subject.lower(), rel.predicate, rel.object.lower())
            if key not in self._relationships:
                self._relationships[key] = rel
                rel_count += 1
            self._file_rels.setdefault(file_id, set()).add(key)

        return {"entities_upserted": ent_count, "relationships_upserted": rel_count}

    def query(
        self,
        query_entities: List[str],
        *,
        max_hops: int = 2,
        top_k: int = 20,
        index_name: Optional[str] = None,
    ) -> GraphQueryResult:
        seed_names = {n.lower() for n in query_entities}

        # Also check aliases
        for name_lower, ent in self._entities.items():
            for alias in ent.aliases:
                if alias.lower() in seed_names:
                    seed_names.add(name_lower)

        # BFS traversal
        visited: set = set()
        frontier = set(seed_names)
        matched_rels: List[Relationship] = []

        for hop in range(max_hops):
            next_frontier: set = set()
            for name in frontier:
                if name in visited:
                    continue
                visited.add(name)
                for key, rel in self._relationships.items():
                    subj_l, pred, obj_l = key
                    if subj_l == name and obj_l not in visited:
                        matched_rels.append(rel)
                        next_frontier.add(obj_l)
                    elif obj_l == name and subj_l not in visited:
                        matched_rels.append(rel)
                        next_frontier.add(subj_l)
            frontier = next_frontier
            if len(matched_rels) >= top_k:
                break

        matched_rels = matched_rels[:top_k]

        # Collect entities
        entity_names = set()
        for rel in matched_rels:
            entity_names.add(rel.subject.lower())
            entity_names.add(rel.object.lower())
        entity_names.update(seed_names & set(self._entities.keys()))

        entities = [
            self._entities[n] for n in entity_names
            if n in self._entities
        ]

        # Collect chunk IDs
        chunk_ids = set()
        for ent in entities:
            cid = ent.properties.get("source_chunk_id", "")
            if cid:
                chunk_ids.add(cid)
        for rel in matched_rels:
            cid = rel.properties.get("source_chunk_id", "")
            if cid:
                chunk_ids.add(cid)

        subgraph_text = GremlinGraphAdapter._format_subgraph(entities, matched_rels)

        return GraphQueryResult(
            entities=entities,
            relationships=matched_rels,
            subgraph_text=subgraph_text,
            source_chunk_ids=sorted(chunk_ids),
        )

    def delete_by_file_id(
        self,
        file_id: str,
        *,
        index_name: Optional[str] = None,
    ) -> int:
        count = 0
        # Delete relationships
        for key in list(self._file_rels.get(file_id, set())):
            if key in self._relationships:
                del self._relationships[key]
                count += 1
        self._file_rels.pop(file_id, None)

        # Delete entities unique to this file
        for name in list(self._file_entities.get(file_id, set())):
            # Check if entity is referenced by other files
            other_files = any(
                name in ents
                for fid, ents in self._file_entities.items()
                if fid != file_id
            )
            if not other_files and name in self._entities:
                del self._entities[name]
                count += 1
        self._file_entities.pop(file_id, None)

        return count

    def get_entity(self, name: str) -> Optional[Entity]:
        return self._entities.get(name.lower())


__all__ = ["GremlinGraphAdapter", "InMemoryGraphStore"]
