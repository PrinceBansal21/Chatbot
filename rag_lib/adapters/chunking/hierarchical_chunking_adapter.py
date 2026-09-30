"""Hierarchical Chunking Adapter — 7-Step Heading-Cascade Algorithm.

Implements a structure-aware chunking strategy that respects document hierarchy:

  1. Parse heading levels (H1, H2, H3) in Markdown
  2. Collect all content under each heading until the next same/higher-level heading
  3. If section content ≤ max_tokens → single chunk (with heading chain as context)
  4. If content > max_tokens → try splitting at sub-headings (H3, H4)
  5. If still > max_tokens → split at paragraph boundaries (double newlines)
  6. If still > max_tokens → split at sentence boundaries (last resort)
  7. Prepend parent heading chain + page_summary to each chunk for contextual embedding

Contextual Embedding:
  Each chunk gets an `embedding_content` field:
      Title + Section Headers + Page Summary + Chunk Text
  used to generate the embedding vector, while `content` stores the original chunk
  text for display in retrieval results.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import hashlib
import logging
import textwrap

from rag_lib.interfaces.chunking_manager import ChunkingManager, Chunk
from rag_lib.utils.page_markers import (
    PAGE_MARKER_RE as _PAGE_MARKER_RE,
    extract_page_number as _extract_page_number_fn,
    extract_all_page_numbers as _extract_all_page_numbers_fn,
    strip_page_markers as _strip_page_markers_fn,
    extract_slide_type_map,
)

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
# _PAGE_MARKER_RE is imported from rag_lib.utils.page_markers
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class HierarchicalChunkingAdapter(ChunkingManager):
    """
    7-step heading-cascade chunking with contextual embedding support.

    Args:
        max_tokens: Maximum tokens per child chunk (default 500).
        min_tokens: Minimum tokens — chunks smaller than this are merged up (default 80).
        overlap_tokens: Token overlap between consecutive sentence-split chunks (default 50).
        parent_max_tokens: Max tokens for parent sections (default 2000).
        include_parents_in_index: Whether to also index parent-level chunks (default True).
        generate_summaries: Whether to generate LLM document summaries (default False).
        summary_config: Config dict for the LLM summary model.
        contextual_embedding: Whether to build ``embedding_content`` from headings + summary (default True).
        max_context_prefix_tokens: Max tokens for the heading/summary prefix (default 120).
    """

    # ── LLM summary prompt (unchanged) ──────────────────────────
    SUMMARY_PROMPT = textwrap.dedent("""\
        Analyze the following document and create a comprehensive summary that captures:
        1. The main topic and purpose of the document
        2. Key facts, figures, and important details
        3. Any actionable items or recommendations
        4. Important entities mentioned (people, organizations, dates, etc.)

        Document:
        {document}

        Provide a detailed summary (3-5 paragraphs) that would help someone understand
        the document without reading it entirely:""")

    def __init__(
        self,
        max_tokens: int = 500,
        min_tokens: int = 80,
        overlap_tokens: int = 50,
        parent_max_tokens: int = 2000,
        include_parents_in_index: bool = True,
        generate_summaries: bool = False,
        summary_config: Optional[Dict[str, Any]] = None,
        contextual_embedding: bool = True,
        max_context_prefix_tokens: int = 120,
        **kwargs,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_max_tokens = parent_max_tokens
        self.include_parents_in_index = include_parents_in_index
        self.generate_summaries = generate_summaries
        self.summary_config = summary_config or {}
        self.contextual_embedding = contextual_embedding
        self.max_context_prefix_tokens = max_context_prefix_tokens

        # Approximate chars-per-token ratio
        self.chars_per_token = 4

        # LLM manager for summaries (lazy-loaded)
        self._llm_manager = None

    # ── helpers ────────────────────────────────────────────────────

    @property
    def llm_manager(self):
        """Lazy-load LLM manager for optional summary generation."""
        if self._llm_manager is None and self.generate_summaries:
            from rag_lib.factory import RAGFactory
            provider = self.summary_config.get("provider") or RAGFactory.get_default_llm_provider()
            from rag_lib.model_config import get_model_spec, resolve_config_provider_key
            spec = get_model_spec("llm", resolve_config_provider_key(provider))
            model_id = self.summary_config.get("model_id") or spec["model_id"]
            region = self.summary_config.get("region") or spec["region"]
            self._llm_manager = RAGFactory.get_llm_manager(
                provider=provider, model_id=model_id, region=region,
            )
        return self._llm_manager

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count: len / 4."""
        return len(text) // 4

    @staticmethod
    def _generate_chunk_id(file_id: str, content: str, order: int) -> str:
        content_hash = hashlib.sha256(content[:500].encode()).hexdigest()[:8]
        return f"{file_id}-{order:05d}-{content_hash}"

    @staticmethod
    def _extract_page_number(text: str) -> Optional[str]:
        """Return the first page-marker value found in *text*, or None."""
        return _extract_page_number_fn(text)

    @staticmethod
    def _extract_all_page_numbers(text: str) -> List[str]:
        """Return all unique page-marker values found in *text*, ordered."""
        return _extract_all_page_numbers_fn(text)

    @staticmethod
    def _strip_page_markers(text: str) -> str:
        """Remove ``<!-- page_number: N -->`` markers from text and clean up extra blank lines."""
        return _strip_page_markers_fn(text)

    # ── 1+2: Parse headings & collect sections ────────────────────

    def _parse_sections(self, markdown: str) -> List[dict]:
        """
        Walk through the markdown and split it into heading-rooted sections.

        Returns a list of dicts::

            {
                "level": int,         # 1-6 (heading depth) or 0 (preamble)
                "heading": str,       # heading text (empty for preamble)
                "body": str,          # all content under this heading
                "heading_chain": [str],  # ancestors + self heading texts
            }
        """
        lines = markdown.split("\n")
        sections: List[dict] = []
        current: Optional[dict] = None
        heading_stack: List[Tuple[int, str]] = []  # (level, heading_text)

        def _flush():
            nonlocal current
            if current is not None:
                current["body"] = "\n".join(current["_lines"]).strip()
                del current["_lines"]
                if current["body"] or current["heading"]:
                    sections.append(current)
            current = None

        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                _flush()
                level = len(m.group(1))
                heading_text = m.group(2).strip()

                # Pop stack until we find a parent with strictly lower level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                chain = [h for _, h in heading_stack]

                current = {
                    "level": level,
                    "heading": heading_text,
                    "heading_chain": chain,
                    "_lines": [],
                }
            else:
                if current is None:
                    # Preamble (before any heading)
                    current = {
                        "level": 0,
                        "heading": "",
                        "heading_chain": [],
                        "_lines": [],
                    }
                current["_lines"].append(line)

        _flush()
        return sections

    # ── 3-6: Cascading split logic ────────────────────────────────

    def _try_split_at_subheadings(self, text: str, parent_level: int) -> Optional[List[str]]:
        """
        Step 4: Split *text* at sub-headings (parent_level+1 .. 6).

        Returns a list of sub-section strings, or ``None`` if no sub-headings found.
        """
        sub_heading_re = re.compile(
            r"^(#{" + str(parent_level + 1) + r",6})\s+", re.MULTILINE
        )
        positions = [m.start() for m in sub_heading_re.finditer(text)]
        if not positions:
            return None

        parts: List[str] = []
        # Content before first sub-heading
        if positions[0] > 0:
            preamble = text[: positions[0]].strip()
            if preamble:
                parts.append(preamble)

        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            part = text[pos:end].strip()
            if part:
                parts.append(part)

        return parts if len(parts) > 1 else None

    def _split_at_paragraphs(self, text: str) -> List[str]:
        """
        Step 5: Split *text* at paragraph boundaries (blank lines).
        Groups consecutive paragraphs to stay under ``max_tokens``.
        """
        paragraphs = re.split(r"\n{2,}", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        groups: List[str] = []
        current_group: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if current_tokens + para_tokens > self.max_tokens and current_group:
                groups.append("\n\n".join(current_group))
                current_group = []
                current_tokens = 0
            current_group.append(para)
            current_tokens += para_tokens

        if current_group:
            groups.append("\n\n".join(current_group))

        return groups

    def _split_at_sentences(self, text: str) -> List[str]:
        """
        Step 6 (last resort): Split *text* at sentence boundaries.
        Uses overlap_tokens for continuity between consecutive chunks.
        """
        sentences = _SENTENCE_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                # Overlap: keep last N tokens worth of sentences
                overlap: List[str] = []
                overlap_tok = 0
                for s in reversed(current):
                    st = self._estimate_tokens(s)
                    if overlap_tok + st > self.overlap_tokens:
                        break
                    overlap.insert(0, s)
                    overlap_tok += st
                current = overlap
                current_tokens = overlap_tok
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _cascade_split(self, text: str, heading_level: int) -> List[str]:
        """
        Steps 3-6: Decide whether *text* needs splitting, and apply the
        cheapest strategy that brings every fragment under ``max_tokens``.

        Returns a list of chunk-text strings.
        """
        # Step 3 — fits already
        if self._estimate_tokens(text) <= self.max_tokens:
            return [text]

        # Step 4 — try sub-headings
        parts = self._try_split_at_subheadings(text, heading_level)
        if parts:
            result: List[str] = []
            for part in parts:
                if self._estimate_tokens(part) <= self.max_tokens:
                    result.append(part)
                else:
                    # Recursively try paragraphs / sentences on this sub-part
                    result.extend(self._split_at_paragraphs_or_sentences(part))
            return result

        # Steps 5 + 6
        return self._split_at_paragraphs_or_sentences(text)

    def _split_at_paragraphs_or_sentences(self, text: str) -> List[str]:
        """Try paragraph split first; fall back to sentence split for oversized pieces."""
        para_chunks = self._split_at_paragraphs(text)

        final: List[str] = []
        for chunk in para_chunks:
            if self._estimate_tokens(chunk) <= self.max_tokens:
                final.append(chunk)
            else:
                # Step 6 — sentence-level
                final.extend(self._split_at_sentences(chunk))
        return final

    # ── 7: Build contextual embedding prefix ──────────────────────

    def _build_embedding_content(
        self,
        chunk_text: str,
        heading_chain: List[str],
        base_metadata: Dict[str, str],
        page_summary: str = "",
    ) -> str:
        """
        Build the contextual text used for embedding:
            Title + Section Headers + Page Summary (≤60 tokens) + Chunk Text

        ``page_summary`` is the per-page summary for the page this chunk
        belongs to.  If unavailable the document-level summary is used as
        fallback.

        The result is stored in ``chunk.embedding_content`` so the embedding
        adapter uses it for vector generation, while ``chunk.content`` keeps the
        original text for storage / display.
        """
        parts: List[str] = []

        # Title
        title = base_metadata.get("title") or base_metadata.get("file_name", "")
        if title:
            parts.append(f"Title: {title}")

        # Section heading chain
        if heading_chain:
            parts.append("Section: " + " > ".join(heading_chain))

        # Page summary (prefer per-page, fall back to document-level)
        summary = page_summary or base_metadata.get("document_summary", "")
        if summary:
            max_chars = 60 * self.chars_per_token  # ~240 chars
            if len(summary) > max_chars:
                summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
            parts.append(f"Page Summary: {summary}")

        # Verify prefix stays within budget
        prefix = "\n".join(parts)
        max_prefix_chars = self.max_context_prefix_tokens * self.chars_per_token
        if len(prefix) > max_prefix_chars:
            prefix = prefix[:max_prefix_chars].rsplit(" ", 1)[0] + "…"

        return f"{prefix}\n\n{chunk_text}" if prefix else chunk_text

    # ── LLM summary (opt-in, unchanged) ──────────────────────────

    def _generate_document_summary(self, markdown: str, file_id: str) -> Optional[str]:
        if not self.generate_summaries or self.llm_manager is None:
            return None
        try:
            max_doc_chars = 100_000
            doc_text = markdown[:max_doc_chars]
            if len(markdown) > max_doc_chars:
                doc_text += "\n\n[Document truncated for summary generation…]"
            prompt = self.SUMMARY_PROMPT.format(document=doc_text)
            response = self.llm_manager.generate(
                prompt=prompt,
                system_prompt="You are an expert document analyst. Create comprehensive, accurate summaries.",
                max_tokens=1024,
                temperature=0.2,
            )
            logger.info(f"Generated summary for {file_id}: {len(response.text)} chars")
            return response.text
        except Exception as e:
            logger.warning(f"Failed to generate summary for {file_id}: {e}")
            return None

    # ── Merge tiny chunks ────────────────────────────────────────

    def _merge_small_chunks(self, texts: List[str]) -> List[str]:
        """Merge consecutive chunks that are below ``min_tokens``."""
        if not texts:
            return texts
        merged: List[str] = []
        buf = texts[0]
        for t in texts[1:]:
            if self._estimate_tokens(buf) < self.min_tokens:
                buf = buf + "\n\n" + t
            else:
                merged.append(buf)
                buf = t
        merged.append(buf)
        return merged

    # ── Main entry point ─────────────────────────────────────────

    def chunk(self, file_id: str, markdown: str, base_metadata: Dict[str, str]) -> List[Chunk]:
        """
        Chunk markdown using the 7-step heading-cascade algorithm.

        Returns:
            List of Chunk objects with ``parent_id`` relationships and
            (optionally) ``embedding_content`` for contextual embedding.
        """
        chunks: List[Chunk] = []
        counter = 0
        parent_counter = 0

        # Captured from the raw markdown before page markers are stripped —
        # PPTX-only, maps page_number -> "title" | "content" (any page position).
        slide_type_map = extract_slide_type_map(markdown)

        # ── Parse per-page summaries from metadata (JSON str → dict) ──
        page_summaries: Dict[str, str] = {}
        raw_page_summaries = base_metadata.get("page_summaries", "")
        if raw_page_summaries:
            try:
                import json
                page_summaries = json.loads(raw_page_summaries)
            except (json.JSONDecodeError, TypeError):
                pass

        # Remove transient field so it doesn't land in every chunk's metadata
        chunk_base_metadata = {
            k: v for k, v in base_metadata.items() if k != "page_summaries"
        }

        # ── optional document summary ──
        if self.generate_summaries:
            summary_text = self._generate_document_summary(markdown, file_id)
            if summary_text:
                summary_meta = chunk_base_metadata.copy()
                summary_meta["chunk_type"] = "summary"
                summary_meta["is_document_summary"] = "true"
                chunks.append(Chunk(
                    chunk_id=f"{file_id}-summary",
                    file_id=file_id,
                    content=summary_text,
                    order=counter,
                    metadata=summary_meta,
                    parent_id=None,
                    level="summary",
                ))
                counter += 1

        # ── Steps 1-2: parse heading sections ──
        sections = self._parse_sections(markdown)

        for section in sections:
            heading_chain: List[str] = section["heading_chain"]
            heading_level: int = section["level"]
            body: str = section["body"]
            heading_text: str = section["heading"]

            # Reconstruct full section text (heading + body)
            if heading_text:
                full_section = f"{'#' * heading_level} {heading_text}\n\n{body}"
            else:
                full_section = body

            if not full_section.strip():
                continue

            # Determine page number(s) from section content
            all_pages = self._extract_all_page_numbers(full_section)
            page_num = all_pages[0] if all_pages else None

            # Strip page markers from content (metadata stores the page number)
            full_section = self._strip_page_markers(full_section)
            if not full_section.strip():
                continue

            # Look up page-level summary for this section
            section_page_summary = page_summaries.get(page_num, "") if page_num else ""

            # ── Parent chunk ──
            parent_id = self._generate_chunk_id(file_id, full_section, parent_counter)
            parent_meta = chunk_base_metadata.copy()
            if page_num:
                parent_meta["page_number"] = page_num
            if page_num in slide_type_map:
                parent_meta["slide_type"] = slide_type_map[page_num]
            if section_page_summary:
                parent_meta["page_summary"] = section_page_summary
            if heading_chain:
                parent_meta["heading_chain"] = " > ".join(heading_chain)
            parent_meta["chunk_type"] = "parent"

            if self.include_parents_in_index:
                # Truncate very long parents (keep first parent_max_tokens)
                parent_content = full_section
                if self._estimate_tokens(parent_content) > self.parent_max_tokens:
                    max_chars = self.parent_max_tokens * self.chars_per_token
                    parent_content = parent_content[:max_chars]

                parent_chunk = Chunk(
                    chunk_id=parent_id,
                    file_id=file_id,
                    content=parent_content,
                    order=counter,
                    metadata=parent_meta,
                    parent_id=None,
                    level="parent",
                )
                if self.contextual_embedding:
                    parent_chunk.embedding_content = self._build_embedding_content(
                        parent_content, heading_chain, chunk_base_metadata,
                        page_summary=section_page_summary,
                    )
                chunks.append(parent_chunk)
                counter += 1

            # ── Steps 3-6: cascade split into children ──
            child_texts = self._cascade_split(body, heading_level)
            child_texts = self._merge_small_chunks(child_texts)

            child_count = 0
            for child_text in child_texts:
                if not child_text.strip():
                    continue

                # Extract page number(s) from child before stripping markers
                child_page = self._extract_page_number(child_text)
                # Strip page markers from child content
                child_text = self._strip_page_markers(child_text)
                if not child_text.strip():
                    continue

                child_id = self._generate_chunk_id(file_id, child_text, counter)
                child_meta = chunk_base_metadata.copy()
                child_meta["chunk_type"] = "child"
                child_meta["parent_chunk_id"] = parent_id
                child_meta["child_index"] = str(child_count)
                if heading_chain:
                    child_meta["heading_chain"] = " > ".join(heading_chain)

                if child_page:
                    child_meta["page_number"] = child_page
                elif page_num:
                    child_meta["page_number"] = page_num

                # Resolve page summary for this child
                effective_page = child_page or page_num
                if effective_page in slide_type_map:
                    child_meta["slide_type"] = slide_type_map[effective_page]
                child_page_summary = (
                    page_summaries.get(effective_page, "") if effective_page else ""
                )
                if child_page_summary:
                    child_meta["page_summary"] = child_page_summary

                child_chunk = Chunk(
                    chunk_id=child_id,
                    file_id=file_id,
                    content=child_text,
                    order=counter,
                    metadata=child_meta,
                    parent_id=parent_id,
                    level="child",
                )

                # ── Step 7: contextual embedding ──
                if self.contextual_embedding:
                    child_chunk.embedding_content = self._build_embedding_content(
                        child_text, heading_chain, chunk_base_metadata,
                        page_summary=child_page_summary,
                    )

                chunks.append(child_chunk)
                counter += 1
                child_count += 1

            parent_counter += 1

        logger.info(
            f"Hierarchical chunking produced {len(chunks)} chunks "
            f"({parent_counter} sections) for {file_id}"
        )
        return chunks


__all__ = ["HierarchicalChunkingAdapter"]
