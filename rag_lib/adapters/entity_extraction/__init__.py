"""LLM-based entity and relationship extractor.

Sends batches of chunks to an LLM and parses the structured JSON response
to produce ``Entity`` and ``Relationship`` objects.

Optimised for cost:
- Batches multiple chunks per LLM call (default batch_size=5)
- Uses a cheaper/faster model (configurable)
- Filters trivial chunks (short headers, TOC lines) before extraction
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from rag_lib.interfaces.graph_manager import (
    Entity,
    EntityExtractor,
    GraphExtractionResult,
    Relationship,
)
from rag_lib.interfaces.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph builder. "
    "Extract ALL meaningful entities and relationships from each text block. "
    "Return ONLY valid JSON — no markdown fences, no commentary."
)

_EXTRACTION_TEMPLATE = """Extract entities and relationships from the text blocks below.

RULES:
1. Entity types: PERSON, ORG, PRODUCT, METRIC, DATE, LOCATION, EVENT, CONCEPT, TECHNOLOGY.
2. Normalise entity names (e.g. "UN" → "United Nations").
3. Keep aliases in the "aliases" array.
4. Predicates should be short verb phrases in snake_case (e.g. "reported_revenue", "acquired", "is_part_of").
5. Each relationship must reference entity names exactly as they appear in "entities".
6. If a block has no meaningful entities, return an empty arrays for that block.

Return a single JSON object:
{{
  "blocks": [
    {{
      "block_id": "<chunk_id>",
      "entities": [
        {{"name": "...", "type": "...", "aliases": ["..."], "description": "..."}}
      ],
      "relationships": [
        {{"subject": "...", "predicate": "...", "object": "...", "evidence": "..."}}
      ]
    }}
  ]
}}

{document_context_section}

{blocks_section}"""

_BLOCK_TEMPLATE = "--- Block {block_id} ---\n{content}\n"

_MIN_CHUNK_LENGTH = 30  # characters – skip trivial chunks


class LLMEntityExtractor(EntityExtractor):
    """Batch entity/relationship extractor powered by an LLM."""

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        extraction_template: Optional[str] = None,
        min_chunk_length: int = _MIN_CHUNK_LENGTH,
        max_workers: int = 4,
    ):
        """
        Args:
            llm_manager: Cloud-neutral low-level LLM adapter.
            max_tokens: Maximum output tokens per LLM call.
            temperature: Sampling temperature (0 for deterministic).
            system_prompt: Override the default system prompt.
            extraction_template: Override the default extraction template.
            min_chunk_length: Skip chunks shorter than this (characters).
            max_workers: Max parallel LLM calls for batch processing.
        """
        self._llm = llm_manager
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._extraction_template = extraction_template or _EXTRACTION_TEMPLATE
        self._min_chunk_length = min_chunk_length
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        chunks: List[Any],
        *,
        batch_size: int = 5,
        document_context: Optional[str] = None,
    ) -> GraphExtractionResult:
        """Extract entities and relationships from chunks.

        Chunks are grouped into batches and processed in parallel.
        """
        # Filter trivial chunks
        meaningful = [
            c for c in chunks
            if len(getattr(c, "content", "") or "") >= self._min_chunk_length
        ]
        if not meaningful:
            return GraphExtractionResult()

        # Create batches
        batches: List[List[Any]] = [
            meaningful[i: i + batch_size]
            for i in range(0, len(meaningful), batch_size)
        ]

        logger.info(
            "Extracting entities from %d chunks in %d batches (batch_size=%d, workers=%d)",
            len(meaningful), len(batches), batch_size, self._max_workers,
        )

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        all_chunk_ids: List[str] = []
        total_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._extract_batch, batch, document_context): batch
                for batch in batches
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                    all_chunk_ids.extend(result.source_chunk_ids)
                    for k in total_usage:
                        total_usage[k] += result.metadata.get(k, 0)
                except Exception:
                    batch = futures[future]
                    chunk_ids = [getattr(c, "chunk_id", "?") for c in batch]
                    logger.exception("Entity extraction failed for batch %s", chunk_ids)

        logger.info(
            "Extraction complete: %d entities, %d relationships from %d chunks",
            len(all_entities), len(all_relationships), len(all_chunk_ids),
        )

        return GraphExtractionResult(
            entities=all_entities,
            relationships=all_relationships,
            source_chunk_ids=all_chunk_ids,
            metadata=total_usage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_batch(
        self,
        chunks: List[Any],
        document_context: Optional[str],
    ) -> GraphExtractionResult:
        """Process a single batch of chunks through the LLM."""
        # Build blocks section
        blocks_lines: List[str] = []
        chunk_ids: List[str] = []
        for chunk in chunks:
            cid = getattr(chunk, "chunk_id", None) or str(uuid.uuid4())
            content = getattr(chunk, "content", "") or ""
            blocks_lines.append(
                _BLOCK_TEMPLATE.format(block_id=cid, content=content.strip())
            )
            chunk_ids.append(cid)

        blocks_section = "\n".join(blocks_lines)

        # Document context section
        doc_ctx = ""
        if document_context:
            doc_ctx = f"Document context (use for disambiguation):\n{document_context}\n"

        prompt = self._extraction_template.format(
            document_context_section=doc_ctx,
            blocks_section=blocks_section,
        )

        # Call LLM
        response = self._llm.generate(
            prompt=prompt,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        # Parse response
        entities, relationships = self._parse_response(response.text, chunk_ids)

        return GraphExtractionResult(
            entities=entities,
            relationships=relationships,
            source_chunk_ids=chunk_ids,
            metadata=response.usage,
        )

    def _parse_response(
        self,
        text: str,
        chunk_ids: List[str],
    ) -> tuple[List[Entity], List[Relationship]]:
        """Parse LLM JSON response into Entity/Relationship objects."""
        # Strip markdown fences if present
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %.200s", text)
            return [], []

        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            # Maybe the LLM returned a flat structure
            blocks = [data]

        entities: List[Entity] = []
        relationships: List[Relationship] = []
        seen_entity_names: set = set()

        for block in blocks:
            block_id = block.get("block_id", "")

            # Parse entities
            for ent_data in block.get("entities", []):
                name = (ent_data.get("name") or "").strip()
                if not name:
                    continue
                etype = (ent_data.get("type") or "CONCEPT").upper()
                aliases = ent_data.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]

                # Dedupe within this extraction run
                if name.lower() not in seen_entity_names:
                    seen_entity_names.add(name.lower())
                    entities.append(Entity(
                        name=name,
                        entity_type=etype,
                        aliases=[a for a in aliases if a],
                        properties={
                            "description": ent_data.get("description", ""),
                            "source_chunk_id": block_id,
                        },
                    ))

            # Parse relationships
            for rel_data in block.get("relationships", []):
                subject = (rel_data.get("subject") or "").strip()
                predicate = (rel_data.get("predicate") or "").strip()
                obj = (rel_data.get("object") or "").strip()
                if not (subject and predicate and obj):
                    continue
                relationships.append(Relationship(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    properties={
                        "evidence": rel_data.get("evidence", ""),
                        "source_chunk_id": block_id,
                    },
                ))

        return entities, relationships


__all__ = ["LLMEntityExtractor"]
