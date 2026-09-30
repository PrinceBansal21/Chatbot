"""Gremlin graph storage adapter.

Stores entities as vertices and relationships as edges in any graph database
that speaks the **Gremlin** traversal language via the ``gremlin_python``
driver.

Supports:
- Upsert semantics (merge on entity name)
- Provenance tracking via ``file_id`` and ``source_chunk_id`` properties
- Sub-graph queries with configurable hop depth
- Delete-by-file_id for re-ingestion
- Namespace isolation via ``index_name`` vertex/edge labels
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from rag_lib.interfaces.graph_manager import (
    Entity,
    GraphExtractionResult,
    GraphQueryResult,
    GraphStore,
    Relationship,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (Gremlin driver)
# ---------------------------------------------------------------------------
try:
    from gremlin_python.driver import client as gremlin_client
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.process.graph_traversal import __
    from gremlin_python.process.traversal import T, P, Cardinality

    _HAS_GREMLIN = True
except ImportError:
    _HAS_GREMLIN = False


class GremlinGraphAdapter(GraphStore):
    """Graph store accessed via the Gremlin traversal language.

    Works against any Gremlin-compatible graph database endpoint.

    Args:
        endpoint: Gremlin websocket endpoint (e.g.
            ``wss://my-cluster.example.com:8182/gremlin``).
            Defaults to the ``GRAPH_DB_ENDPOINT`` environment variable.
        default_label: Default vertex/edge label prefix when ``index_name`` is not supplied.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_label: str = "kg",
    ):
        if not _HAS_GREMLIN:
            raise ImportError(
                "gremlin_python is required for GremlinGraphAdapter. "
                "Install with: pip install gremlinpython"
            )
        self._endpoint = endpoint or os.environ.get("GRAPH_DB_ENDPOINT", "")
        if not self._endpoint:
            raise RuntimeError("GRAPH_DB_ENDPOINT must be set for GremlinGraphAdapter.")
        self._default_label = default_label
        self._connection = None
        self._g = None

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    @property
    def g(self):
        """Lazy-initialised Gremlin traversal source."""
        if self._g is None:
            self._connection = DriverRemoteConnection(self._endpoint, "g")
            self._g = traversal().withRemote(self._connection)
        return self._g

    def close(self):
        """Close the Gremlin connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._g = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        extraction: GraphExtractionResult,
        *,
        file_id: str,
        index_name: Optional[str] = None,
    ) -> Dict[str, int]:
        label = index_name or self._default_label
        ent_label = f"{label}_entity"
        rel_label = f"{label}_rel"

        ent_count = 0
        rel_count = 0

        # Upsert entities
        for entity in extraction.entities:
            try:
                self._upsert_entity(entity, ent_label, file_id)
                ent_count += 1
            except Exception:
                logger.exception("Failed to upsert entity: %s", entity.name)

        # Upsert relationships
        for rel in extraction.relationships:
            try:
                self._upsert_relationship(rel, ent_label, rel_label, file_id)
                rel_count += 1
            except Exception:
                logger.exception(
                    "Failed to upsert relationship: %s -[%s]-> %s",
                    rel.subject, rel.predicate, rel.object,
                )

        logger.info(
            "Graph upsert complete: %d entities, %d relationships (label=%s, file_id=%s)",
            ent_count, rel_count, label, file_id,
        )
        return {"entities_upserted": ent_count, "relationships_upserted": rel_count}

    def query(
        self,
        query_entities: List[str],
        *,
        max_hops: int = 2,
        top_k: int = 20,
        index_name: Optional[str] = None,
    ) -> GraphQueryResult:
        label = index_name or self._default_label
        ent_label = f"{label}_entity"
        rel_label = f"{label}_rel"

        if not query_entities:
            return GraphQueryResult()

        # Find seed vertices matching query entities (case-insensitive)
        seed_names = [n.lower() for n in query_entities]

        try:
            # Find seed vertex IDs
            seed_vertices = (
                self.g.V()
                .hasLabel(ent_label)
                .has("name_lower", P.within(seed_names))
                .valueMap(True)
                .toList()
            )

            if not seed_vertices:
                # Try alias matching
                seed_vertices = (
                    self.g.V()
                    .hasLabel(ent_label)
                    .has("aliases_lower", P.within(seed_names))
                    .valueMap(True)
                    .toList()
                )

            if not seed_vertices:
                return GraphQueryResult()

            # Get seed vertex IDs
            seed_ids = [v[T.id] for v in seed_vertices]

            # Traverse N hops from seeds
            traversal_result = (
                self.g.V(seed_ids)
                .repeat(__.bothE(rel_label).otherV().simplePath())
                .times(max_hops)
                .emit()
                .path()
                .by(__.valueMap(True))
                .limit(top_k * 3)  # over-fetch to account for dedup
                .toList()
            )

            # Parse traversal results
            entities, relationships, chunk_ids = self._parse_traversal(
                seed_vertices, traversal_result, top_k
            )

            # Build human-readable subgraph text
            subgraph_text = self._format_subgraph(entities, relationships)

            return GraphQueryResult(
                entities=entities,
                relationships=relationships,
                subgraph_text=subgraph_text,
                source_chunk_ids=chunk_ids,
            )

        except Exception:
            logger.exception("Graph query failed for entities: %s", query_entities)
            return GraphQueryResult()

    def delete_by_file_id(
        self,
        file_id: str,
        *,
        index_name: Optional[str] = None,
    ) -> int:
        label = index_name or self._default_label
        ent_label = f"{label}_entity"
        rel_label = f"{label}_rel"

        try:
            # Delete edges from this file first
            edge_count = (
                self.g.E()
                .hasLabel(rel_label)
                .has("file_id", file_id)
                .count()
                .next()
            )
            self.g.E().hasLabel(rel_label).has("file_id", file_id).drop().iterate()

            # Delete vertices that ONLY came from this file
            # (don't delete vertices that also appear in other documents)
            vertex_count = (
                self.g.V()
                .hasLabel(ent_label)
                .has("file_ids", file_id)
                .where(__.values("file_ids").count().is_(1))
                .count()
                .next()
            )
            self.g.V().hasLabel(ent_label).has("file_ids", file_id).where(
                __.values("file_ids").count().is_(1)
            ).drop().iterate()

            # Remove file_id from multi-source vertices
            self.g.V().hasLabel(ent_label).has("file_ids", file_id).properties(
                "file_ids"
            ).hasValue(file_id).drop().iterate()

            total = edge_count + vertex_count
            logger.info(
                "Deleted %d graph elements for file_id=%s (%d edges, %d vertices)",
                total, file_id, edge_count, vertex_count,
            )
            return total

        except Exception:
            logger.exception("Failed to delete graph data for file_id=%s", file_id)
            return 0

    def get_entity(self, name: str) -> Optional[Entity]:
        try:
            results = (
                self.g.V()
                .has("name_lower", name.lower())
                .valueMap(True)
                .toList()
            )
            if results:
                v = results[0]
                return Entity(
                    name=self._first(v.get("name", [name])),
                    entity_type=self._first(v.get("entity_type", ["CONCEPT"])),
                    aliases=v.get("aliases", []),
                )
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_entity(self, entity: Entity, label: str, file_id: str):
        """Upsert a single entity vertex."""
        name_lower = entity.name.lower()
        aliases_lower = [a.lower() for a in entity.aliases]

        # Merge on name_lower
        t = (
            self.g.V()
            .hasLabel(label)
            .has("name_lower", name_lower)
            .fold()
            .coalesce(
                __.unfold(),
                __.addV(label).property("name_lower", name_lower),
            )
            .property(Cardinality.single, "name", entity.name)
            .property(Cardinality.single, "entity_type", entity.entity_type)
            .property(Cardinality.set_, "file_ids", file_id)
        )

        # Add aliases
        for alias in entity.aliases:
            t = t.property(Cardinality.set_, "aliases", alias)
        for alias_l in aliases_lower:
            t = t.property(Cardinality.set_, "aliases_lower", alias_l)

        # Add description
        desc = entity.properties.get("description", "")
        if desc:
            t = t.property(Cardinality.single, "description", desc)

        # Add source chunk
        src_chunk = entity.properties.get("source_chunk_id", "")
        if src_chunk:
            t = t.property(Cardinality.set_, "source_chunk_ids", src_chunk)

        t.iterate()

    def _upsert_relationship(
        self, rel: Relationship, ent_label: str, rel_label: str, file_id: str
    ):
        """Upsert a single relationship edge."""
        subj_lower = rel.subject.lower()
        obj_lower = rel.object.lower()

        # Ensure both vertices exist
        self.g.V().hasLabel(ent_label).has("name_lower", subj_lower).fold().coalesce(
            __.unfold(),
            __.addV(ent_label)
            .property("name_lower", subj_lower)
            .property("name", rel.subject),
        ).iterate()

        self.g.V().hasLabel(ent_label).has("name_lower", obj_lower).fold().coalesce(
            __.unfold(),
            __.addV(ent_label)
            .property("name_lower", obj_lower)
            .property("name", rel.object),
        ).iterate()

        # Add edge (check if exists first)
        self.g.V().hasLabel(ent_label).has("name_lower", subj_lower).as_("s").V().hasLabel(
            ent_label
        ).has("name_lower", obj_lower).as_("o").coalesce(
            __.select("s")
            .outE(rel_label)
            .where(__.inV().as_("o"))
            .has("predicate", rel.predicate),
            __.select("s").addE(rel_label).to(__.select("o")),
        ).property(
            "predicate", rel.predicate
        ).property(
            "file_id", file_id
        ).iterate()

        # Add evidence
        evidence = rel.properties.get("evidence", "")
        if evidence:
            self.g.V().hasLabel(ent_label).has("name_lower", subj_lower).outE(
                rel_label
            ).has("predicate", rel.predicate).where(
                __.inV().has("name_lower", obj_lower)
            ).property(
                "evidence", evidence
            ).iterate()

        # Add source chunk
        src_chunk = rel.properties.get("source_chunk_id", "")
        if src_chunk:
            self.g.V().hasLabel(ent_label).has("name_lower", subj_lower).outE(
                rel_label
            ).has("predicate", rel.predicate).where(
                __.inV().has("name_lower", obj_lower)
            ).property(
                Cardinality.set_, "source_chunk_ids", src_chunk
            ).iterate()

    def _parse_traversal(
        self,
        seed_vertices: list,
        paths: list,
        top_k: int,
    ) -> tuple[List[Entity], List[Relationship], List[str]]:
        """Parse Gremlin path results into entities and relationships."""
        entities_map: Dict[str, Entity] = {}
        relationships: List[Relationship] = []
        chunk_ids: set = set()

        # Add seed vertices
        for v in seed_vertices:
            ent = self._vertex_to_entity(v)
            entities_map[ent.name.lower()] = ent
            for cid in v.get("source_chunk_ids", []):
                chunk_ids.add(cid)

        # Process paths
        rel_count = 0
        for path in paths:
            if rel_count >= top_k:
                break
            objects = path.objects if hasattr(path, "objects") else path
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if T.label in obj and "entity" in str(obj.get(T.label, "")):
                    ent = self._vertex_to_entity(obj)
                    entities_map[ent.name.lower()] = ent
                    for cid in obj.get("source_chunk_ids", []):
                        chunk_ids.add(cid)
                elif "predicate" in obj:
                    # This is an edge
                    rel = Relationship(
                        subject=self._first(obj.get("outV_name", [""])),
                        predicate=self._first(obj.get("predicate", [""])),
                        object=self._first(obj.get("inV_name", [""])),
                        properties={
                            "evidence": self._first(obj.get("evidence", [""])),
                        },
                    )
                    if rel.subject and rel.object:
                        relationships.append(rel)
                        rel_count += 1

        return list(entities_map.values()), relationships[:top_k], sorted(chunk_ids)

    def _vertex_to_entity(self, v: dict) -> Entity:
        """Convert a Gremlin valueMap result to an Entity."""
        return Entity(
            name=self._first(v.get("name", ["Unknown"])),
            entity_type=self._first(v.get("entity_type", ["CONCEPT"])),
            aliases=v.get("aliases", []),
            properties={
                "description": self._first(v.get("description", [""])),
            },
        )

    @staticmethod
    def _first(val):
        """Get first element if list, else return as-is."""
        if isinstance(val, list):
            return val[0] if val else ""
        return val

    @staticmethod
    def _format_subgraph(
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> str:
        """Build a human-readable summary of the sub-graph for LLM context."""
        lines = []

        if entities:
            lines.append("=== Entities ===")
            for ent in entities:
                desc = ent.properties.get("description", "")
                line = f"- {ent.name} ({ent.entity_type})"
                if desc:
                    line += f": {desc}"
                if ent.aliases:
                    line += f" [aliases: {', '.join(ent.aliases)}]"
                lines.append(line)

        if relationships:
            lines.append("\n=== Relationships ===")
            for rel in relationships:
                evidence = rel.properties.get("evidence", "")
                line = f"- {rel.subject} --[{rel.predicate}]--> {rel.object}"
                if evidence:
                    line += f'  (evidence: "{evidence}")'
                lines.append(line)

        return "\n".join(lines) if lines else ""


__all__ = ["GremlinGraphAdapter"]
