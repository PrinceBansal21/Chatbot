"""Entity resolution / deduplication.

Normalises entity names across extraction batches so that different surface
forms (e.g. "UN", "United Nations", "united nations") resolve to a
single canonical entity.  Relationships are remapped accordingly.

Two strategies are provided:
- **RuleBasedEntityResolver** (default) — fast, no LLM cost, uses alias
  matching + fuzzy string similarity.
- An LLM-based resolver can be built on top by subclassing ``EntityResolver``.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from rag_lib.interfaces.graph_manager import (
    Entity,
    EntityResolver,
    GraphExtractionResult,
    Relationship,
)

logger = logging.getLogger(__name__)


def _normalise(name: str) -> str:
    """Lower-case, strip non-alphanum for comparison."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


class RuleBasedEntityResolver(EntityResolver):
    """Fast rule-based entity resolution.

    Resolution pipeline:
    1. Build alias map from existing + new entities.
    2. For each new entity, check exact alias match → merge.
    3. Fallback to fuzzy SequenceMatcher similarity above threshold.
    4. Remap relationship subjects/objects to canonical names.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        merge_aliases: bool = True,
    ):
        """
        Args:
            similarity_threshold: Minimum SequenceMatcher ratio to consider
                two entity names as the same entity (0–1).
            merge_aliases: If True, merge aliases when entities are merged.
        """
        self._threshold = similarity_threshold
        self._merge_aliases = merge_aliases

    def resolve(
        self,
        extraction: GraphExtractionResult,
        *,
        existing_entities: Optional[List[Entity]] = None,
    ) -> GraphExtractionResult:
        existing = list(existing_entities or [])

        # Index: normalised_name → canonical Entity
        canon_map: Dict[str, Entity] = {}
        for ent in existing:
            canon_map[_normalise(ent.name)] = ent
            for alias in ent.aliases:
                canon_map[_normalise(alias)] = ent

        # Phase 1: exact alias matching + fuzzy merge
        resolved_entities: List[Entity] = list(existing)
        name_remap: Dict[str, str] = {}  # old name → canonical name

        for ent in extraction.entities:
            norm = _normalise(ent.name)
            matched_entity = self._find_match(norm, ent, canon_map)

            if matched_entity is not None:
                # Merge into existing
                name_remap[ent.name] = matched_entity.name
                if self._merge_aliases:
                    new_aliases = set(matched_entity.aliases)
                    new_aliases.add(ent.name)
                    new_aliases.update(ent.aliases)
                    new_aliases.discard(matched_entity.name)
                    matched_entity.aliases = sorted(new_aliases)
                # Update alias index
                for alias in ent.aliases:
                    canon_map[_normalise(alias)] = matched_entity
            else:
                # New entity
                canon_map[norm] = ent
                for alias in ent.aliases:
                    canon_map[_normalise(alias)] = ent
                resolved_entities.append(ent)
                name_remap[ent.name] = ent.name

        # Phase 2: remap relationships
        resolved_rels: List[Relationship] = []
        seen_rels: Set[Tuple[str, str, str]] = set()

        for rel in extraction.relationships:
            subj = name_remap.get(rel.subject, rel.subject)
            obj = name_remap.get(rel.object, rel.object)
            pred = rel.predicate
            key = (subj.lower(), pred.lower(), obj.lower())
            if key not in seen_rels:
                seen_rels.add(key)
                resolved_rels.append(Relationship(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    properties=rel.properties,
                ))

        # Deduplicate resolved entities (by canonical name)
        deduped: Dict[str, Entity] = {}
        for ent in resolved_entities:
            key = ent.name.lower()
            if key in deduped:
                existing_ent = deduped[key]
                if self._merge_aliases:
                    merged = set(existing_ent.aliases)
                    merged.update(ent.aliases)
                    merged.discard(existing_ent.name)
                    existing_ent.aliases = sorted(merged)
            else:
                deduped[key] = ent

        logger.info(
            "Entity resolution: %d → %d entities, %d → %d relationships",
            len(extraction.entities), len(deduped),
            len(extraction.relationships), len(resolved_rels),
        )

        return GraphExtractionResult(
            entities=list(deduped.values()),
            relationships=resolved_rels,
            source_chunk_ids=extraction.source_chunk_ids,
            metadata=extraction.metadata,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_match(
        self,
        normalised: str,
        entity: Entity,
        canon_map: Dict[str, Entity],
    ) -> Optional[Entity]:
        """Try exact alias match, then fuzzy SequenceMatcher."""
        # Exact match by normalized name or aliases
        if normalised in canon_map:
            return canon_map[normalised]
        for alias in entity.aliases:
            norm_alias = _normalise(alias)
            if norm_alias in canon_map:
                return canon_map[norm_alias]

        # Fuzzy match against all canonical names
        best_match: Optional[Entity] = None
        best_score = 0.0
        for key, canon_ent in canon_map.items():
            score = SequenceMatcher(None, normalised, key).ratio()
            if score > best_score and score >= self._threshold:
                best_score = score
                best_match = canon_ent

        return best_match


__all__ = ["RuleBasedEntityResolver"]
