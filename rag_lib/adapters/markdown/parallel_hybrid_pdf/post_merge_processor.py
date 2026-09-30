"""
Post-Merge Processor for Parallel Hybrid PDF Adapter

Addresses quality issues that arise from naive page concatenation when
pages are processed by different adapters (Docling vs Vision LLM):

1. **Heading normalization** – infers heading hierarchy when all headings
   are flat (same level) using text patterns (ALL-CAPS, numbered sections),
   and closes hierarchy gaps for pages with varied heading levels.
2. **Duplicate heading removal** – strips identical consecutive headings
   that commonly appear at page boundaries.
3. **Header/footer deduplication** – detects and removes repeated
   header/footer lines across pages (e.g. company name, page numbers,
   document titles).
4. **Cross-page sentence continuity** – merges sentence fragments split
   across page boundaries.
5. **Style normalization** – harmonizes minor formatting differences
   between Docling and Vision LLM output.
6. **Whitespace cleanup** – collapses excessive blank lines.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from rag_lib.adapters.markdown.parallel_hybrid_pdf.heading_scanner import HeadingMap

LOGGER = logging.getLogger(__name__)

# Regex patterns
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PAGE_MARKER_RE = re.compile(r"<!--\s*page_number:\s*(\d+)\s*-->")
_SOURCE_MARKER_RE = re.compile(r"<!--\s*source:\s*(\w+)\s*-->")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Pattern for numbered section headings (e.g. "01 CONSULTING")
_NUMBERED_SECTION_RE = re.compile(r"^\d{1,2}\s+[A-Z]")

# Pattern for metric-value headings (84%, $10.5 bn, ₹131.4 bn, 233,346, etc.)
_METRIC_HEADING_RE = re.compile(
    r"^(#{1,6})\s+"                             # heading prefix
    r"("
    r"[₹$€£¥]?\s*"                              # optional currency symbol
    r"\d[\d,]*\.?\d*"                            # number
    r"\s*"
    r"(%|bn|mn|cr|k|m|x|bps|pp|GW|MW|hrs|days|lakh|crore)?"  # optional unit
    r"\+?"                                       # optional trailing +
    r")"
    r"\s*$",
    re.IGNORECASE,
)


class PostMergeProcessor:
    """
    Cleans and normalises the merged markdown produced by
    ``ParallelHybridPdfAdapter`` after individual page results
    have been concatenated.
    """

    def __init__(
        self,
        *,
        normalize_headings: bool = True,
        deduplicate_headings: bool = True,
        remove_headers_footers: bool = True,
        fix_cross_page_sentences: bool = True,
        normalize_style: bool = True,
        header_footer_threshold: float = 0.5,
        fix_misplaced_tables: bool = True,
        fix_list_item_numbering: bool = True,
    ):
        """
        Args:
            normalize_headings: Fix heading-hierarchy gaps and demote H1→H2.
            deduplicate_headings: Remove identical consecutive headings at
                page boundaries.
            remove_headers_footers: Detect and remove repeated header/footer
                lines that appear on most pages.
            fix_cross_page_sentences: Merge sentence fragments split across
                page breaks.
            normalize_style: Harmonize minor formatting differences between
                adapters.
            header_footer_threshold: Fraction of pages a line must appear on
                to be considered a header/footer (0.0–1.0, default 0.5).
            fix_misplaced_tables: Reattach a table to the numbered list item
                that introduces it (e.g. ``"2. ... provided below:"``) when
                Docling's reading-order model has pushed the table past one
                or more subsequent list items.
            fix_list_item_numbering: Correct a top-level numbered item whose
                digit was misread by Docling (e.g. a genuine item "8." OCR'd
                as "6.") by detecting an out-of-sequence repeat of an
                already-seen number and renumbering it to the expected next
                value in the document-wide sequence.
        """
        self.normalize_headings = normalize_headings
        self.deduplicate_headings = deduplicate_headings
        self.remove_headers_footers = remove_headers_footers
        self.fix_cross_page_sentences = fix_cross_page_sentences
        self.normalize_style = normalize_style
        self.header_footer_threshold = header_footer_threshold
        self.fix_misplaced_tables = fix_misplaced_tables
        self.fix_list_item_numbering = fix_list_item_numbering

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, markdown: str, *, heading_map: Optional[HeadingMap] = None) -> str:
        """
        Apply all enabled post-merge passes to *markdown*.

        Args:
            markdown: The merged markdown text.
            heading_map: Optional heading map produced by
                :func:`heading_scanner.scan_heading_structure`.  When
                supplied the heading-normalisation pass uses font-metric
                data instead of text-pattern heuristics.

        The order of operations is intentional:
          1. Header/footer removal  (operates on per-page blocks)
          2. Cross-page sentence fix (operates on page boundaries)
          3. Style normalization     (local line-level fixes)
          4. Misplaced-table fix     (reattach tables to their list item)
          5. List-item numbering fix (correct OCR'd out-of-sequence numbers)
          6. Heading normalization   (global hierarchy pass)
          7. Heading deduplication   (consecutive duplicate removal)
          8. Whitespace cleanup      (final cosmetic pass)
        """
        if not markdown or not markdown.strip():
            return markdown

        # Page-level passes: parse pages once, avoiding redundant splits
        if self.remove_headers_footers or self.fix_cross_page_sentences or self.fix_misplaced_tables:
            pages = self._split_into_pages(markdown)
            if self.remove_headers_footers:
                pages = self._remove_hf_from_pages(pages)
            if self.fix_cross_page_sentences:
                pages = self._fix_sentences_from_pages(pages)
            if self.fix_misplaced_tables:
                pages = self._fix_misplaced_tables_from_pages(pages)
            markdown = self._pages_to_markdown(pages)

        if self.fix_list_item_numbering:
            markdown = self._fix_list_item_numbering(markdown)

        if self.normalize_style:
            markdown = self._normalize_style(markdown)

        if self.normalize_headings:
            markdown = self._normalize_headings(markdown, heading_map=heading_map)

        # Convert metric-value headings (e.g. ### 84%) to bullet points
        # merged with the following context line.  This runs *after*
        # heading normalization so it catches both mapped and unmapped
        # metric headings.
        markdown = self._convert_metric_headings(markdown)

        if self.deduplicate_headings:
            markdown = self._deduplicate_boundary_headings(markdown)

        # Final whitespace cleanup
        markdown = _MULTI_NL_RE.sub("\n\n", markdown)

        return markdown.strip()

    # ------------------------------------------------------------------
    # 1. Header / footer deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _pages_to_markdown(
        pages: List[Tuple[int, str, str]],
    ) -> str:
        """Serialise a list of ``(page_num, source, content)`` tuples
        back into the merged markdown format with page/source markers."""
        parts: List[str] = []
        for page_num, source, content in pages:
            parts.append(f"<!-- page_number: {page_num} -->")
            if source:
                parts.append(f"<!-- source: {source} -->")
            if content.strip():
                parts.append(content.strip())
        return "\n\n".join(parts)

    def _remove_hf_from_pages(
        self,
        pages: List[Tuple[int, str, str]],
    ) -> List[Tuple[int, str, str]]:
        """Remove repeated header/footer lines, operating on pre-parsed
        page tuples to avoid an extra ``_split_into_pages`` call."""
        if len(pages) < 3:
            return pages

        candidate_counts: Counter = Counter()
        edge_size = 3

        for _pn, _src, content in pages:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if not lines:
                continue
            edge_lines = set(lines[:edge_size] + lines[-edge_size:])
            for line in edge_lines:
                normalised = self._normalise_for_hf_detection(line)
                if normalised:
                    candidate_counts[normalised] += 1

        threshold = max(2, int(len(pages) * self.header_footer_threshold))
        repeated = {
            line for line, count in candidate_counts.items()
            if count >= threshold
        }
        if not repeated:
            return pages

        LOGGER.info(
            "Detected %d header/footer patterns across %d pages",
            len(repeated), len(pages),
        )
        LOGGER.debug("Header/footer patterns: %s", repeated)

        cleaned: List[Tuple[int, str, str]] = []
        for page_num, source, content in pages:
            cleaned_lines: List[str] = []
            for line in content.split("\n"):
                normalised = self._normalise_for_hf_detection(line.strip())
                if normalised and normalised in repeated:
                    continue
                cleaned_lines.append(line)
            cleaned.append((page_num, source, "\n".join(cleaned_lines).strip()))

        return cleaned

    def _fix_sentences_from_pages(
        self,
        pages: List[Tuple[int, str, str]],
    ) -> List[Tuple[int, str, str]]:
        """Merge cross-page sentence fragments, operating on pre-parsed
        page tuples to avoid an extra ``_split_into_pages`` call."""
        if len(pages) < 2:
            return pages

        result: List[Tuple[int, str, str]] = [pages[0]]

        for idx in range(1, len(pages)):
            page_num, source, content = pages[idx]
            prev_pn, prev_src, prev_content = result[-1]

            prev_lines = [l for l in prev_content.split("\n") if l.strip()]
            curr_lines = [l for l in content.split("\n") if l.strip()]

            if prev_lines and curr_lines:
                last_line = prev_lines[-1].rstrip()
                first_line = curr_lines[0]

                if self._is_incomplete_sentence(last_line) and self._is_continuation(first_line):
                    LOGGER.debug(
                        "Merging cross-page sentence: p%d '...%s' + p%d '%s...'",
                        prev_pn, last_line[-40:], page_num, first_line[:40],
                    )
                    merged_line = last_line + " " + first_line.lstrip()

                    # Update previous page content
                    prev_content_lines = prev_content.split("\n")
                    for i in range(len(prev_content_lines) - 1, -1, -1):
                        if prev_content_lines[i].strip():
                            prev_content_lines[i] = merged_line
                            break
                    result[-1] = (prev_pn, prev_src, "\n".join(prev_content_lines))

                    remaining = "\n".join(curr_lines[1:]).strip()
                    result.append((page_num, source, remaining))
                    continue

            result.append((page_num, source, content))

        return result

    # ------------------------------------------------------------------
    # Misplaced-table fix (Docling reading-order bug)
    # ------------------------------------------------------------------

    # A top-level numbered list item, e.g. "2. Composition of CSR Committee: ..."
    _LIST_ITEM_START_RE = re.compile(r"^\d{1,2}\.\s")

    # A lettered sub-item, e.g. "e) CSR amount spent ..." or "- f) Excess ..."
    # (Docling commonly renders these as dash-prefixed bullets.)
    _SUB_ITEM_START_RE = re.compile(r"^-?\s*[a-z]\)\s")

    # A chunk that is *only* bare numeral fragments, e.g. "5. 8." — this is
    # Docling misreading a table's own numeric column headers (1, 2, 3, ...)
    # as list-item markers and bleeding them into the surrounding text. It
    # is noise, not a real list item, and must be dropped before anchor
    # detection so it doesn't break chunk boundaries.
    _NUMERAL_NOISE_RE = re.compile(r"^(?:\d{1,2}\.\s*)+$")

    def _fix_misplaced_tables_from_pages(
        self,
        pages: List[Tuple[int, str, str]],
    ) -> List[Tuple[int, str, str]]:
        """
        Reattach a table to the numbered/lettered list item that introduces it.

        Docling's reading-order model sometimes emits a table *after* one
        or more subsequent list items instead of directly under the item
        that promises it (e.g. "2. ... provided below:"). This pass
        detects that pattern *within a single page's content* and moves
        the table back to immediately follow its introducing item.

        Heuristic (deliberately narrow, to avoid false positives):
          - A chunk (blank-line-separated block, further split at each
            ``N. `` or ``a) `` marker) is a "table" if it starts with ``|``.
          - An item is an "anchor" if its text ends with ``:`` (i.e. it
            promises content "below").
          - Bare numeral-fragment chunks (e.g. "5. 8.") produced by Docling
            misreading a table's own column headers are dropped outright.
          - Anchors without an immediately-following table, and tables not
            immediately following an anchor, are paired up in the order
            they appear on the page (1st orphan anchor <-> 1st orphan
            table, 2nd <-> 2nd, ...) and each orphan table is moved to sit
            directly after its paired anchor. This correctly handles pages
            where several anchors (e.g. "e)", "f)", "7.") appear together
            before their respective tables all appear later, in the same
            relative order.
        """
        result: List[Tuple[int, str, str]] = []

        for page_num, source, content in pages:
            if "|" not in content:
                result.append((page_num, source, content))
                continue

            new_content = self._reattach_tables_to_anchors(content)
            result.append((page_num, source, new_content))

        return result

    @classmethod
    def _reattach_tables_to_anchors(cls, content: str) -> str:
        """Move misplaced table chunks back under the list item that
        introduces them. Operates on a single page's markdown content."""
        raw_chunks = cls._split_into_units(content)

        def is_numeral_noise(chunk: str) -> bool:
            return bool(cls._NUMERAL_NOISE_RE.match(chunk.strip()))

        chunks = [c for c in raw_chunks if not is_numeral_noise(c)]
        noise_removed = len(chunks) != len(raw_chunks)

        if len(chunks) < 3:
            return "\n\n".join(chunks) if noise_removed else content

        def is_table(chunk: str) -> bool:
            stripped = chunk.strip()
            return stripped.startswith("|")

        def is_anchor_item(chunk: str) -> bool:
            stripped = chunk.strip()
            if is_table(chunk):
                return False
            if not (cls._LIST_ITEM_START_RE.match(stripped) or cls._SUB_ITEM_START_RE.match(stripped)):
                return False
            return stripped.rstrip().endswith(":")

        n = len(chunks)

        # Pair every anchor with a table in strict document order (1st
        # anchor <-> 1st table, 2nd <-> 2nd, ...), regardless of where each
        # table currently sits. This is safer than treating a table that
        # merely sits next to *some* anchor as "already correct", because
        # Docling's bug can shift a table so it lands beside the *wrong*
        # anchor (e.g. a table meant for "e)" ends up glued to a later
        # "7." instead) -- which would otherwise look superficially fine.
        anchor_indices = [idx for idx in range(n) if is_anchor_item(chunks[idx])]
        table_indices = [idx for idx in range(n) if is_table(chunks[idx])]

        pair_count = min(len(anchor_indices), len(table_indices))
        anchor_to_table: Dict[int, int] = {
            anchor_indices[k]: table_indices[k] for k in range(pair_count)
        }
        consumed_tables = set(anchor_to_table.values())

        if not anchor_to_table:
            return "\n\n".join(chunks) if noise_removed else content

        for a_idx, t_idx in anchor_to_table.items():
            if t_idx != a_idx + 1:
                LOGGER.debug(
                    "Reattaching misplaced table (chunk %d) to list item anchor (chunk %d)",
                    t_idx, a_idx,
                )

        new_chunks: List[str] = []
        for idx, chunk in enumerate(chunks):
            if idx in consumed_tables:
                continue
            new_chunks.append(chunk)
            if idx in anchor_to_table:
                new_chunks.append(chunks[anchor_to_table[idx]])

        result = "\n\n".join(new_chunks)
        original = "\n\n".join(chunks)
        return result if (noise_removed or result != original) else content

    @classmethod
    def _split_into_units(cls, content: str) -> List[str]:
        """
        Split page content into logical units for the misplaced-table fix.

        Splits first on blank lines (paragraph boundaries), then further
        splits each paragraph at every internal top-level numbered-item
        marker (e.g. ``"2. "``) and at every internal table-row marker
        (a line starting with ``"|"``). This is needed because Docling
        commonly emits consecutive numbered list items (``1.``, ``2.``,
        ``3.``) with no blank line between them, which would otherwise
        collapse into a single paragraph chunk and hide the anchor item.
        It also sometimes glues a table directly onto the preceding list
        item's text with only a single newline (no blank line), which
        would otherwise hide the table inside that item's chunk and
        prevent it from being recognised (and reattached) as a table.
        """
        units: List[str] = []
        for para in content.split("\n\n"):
            lines = para.split("\n")
            current: List[str] = [lines[0]] if lines else []
            in_table = bool(current) and current[0].strip().startswith("|")
            for line in lines[1:]:
                stripped_line = line.strip()
                starts_table_row = stripped_line.startswith("|")
                if starts_table_row and not in_table:
                    # A table row starting mid-paragraph (no blank line
                    # separating it from the preceding text) begins a new
                    # unit so it can be detected as its own table chunk.
                    if current:
                        units.append("\n".join(current))
                    current = [line]
                    in_table = True
                elif cls._LIST_ITEM_START_RE.match(stripped_line) or cls._SUB_ITEM_START_RE.match(stripped_line):
                    if current:
                        units.append("\n".join(current))
                    current = [line]
                    in_table = False
                else:
                    current.append(line)
                    if not starts_table_row:
                        in_table = False
            if current:
                units.append("\n".join(current))
        return units

    # A top-level numbered item with its number captured separately, e.g.
    # "8. Whether any capital assets ..." -> group(1)="8", group(2)=". ".
    _LIST_ITEM_NUM_RE = re.compile(r"^(\d{1,2})(\.\s)")

    def _fix_list_item_numbering(self, markdown: str) -> str:
        """
        Correct a top-level numbered item whose leading digit Docling
        misread (e.g. a genuine item "8." OCR'd as "6.").

        Top-level items in these documents form a strictly increasing
        sequence (1, 2, 3, ...). If a later item's number is *not* greater
        than the highest number already seen -- i.e. it repeats an earlier
        number instead of continuing the sequence -- that is almost
        certainly a misread digit rather than a legitimate restart, so it
        is rewritten to the expected next number in the sequence.

        The sequence is reset at each Markdown heading, since a new
        section/annexure may legitimately restart numbering from 1.
        """
        lines = markdown.split("\n")
        expected_next: Optional[int] = None
        changed = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                expected_next = None
                continue

            match = self._LIST_ITEM_NUM_RE.match(stripped)
            if not match:
                continue

            num = int(match.group(1))
            if expected_next is None:
                expected_next = num + 1
                continue

            if num >= expected_next:
                expected_next = num + 1
                continue

            # num < expected_next: an out-of-sequence repeat of an
            # already-used number. Renumber to the expected next value.
            correct = expected_next
            leading_ws_len = len(line) - len(line.lstrip())
            new_line = (
                line[:leading_ws_len]
                + str(correct)
                + match.group(2)
                + line[leading_ws_len + match.end():]
            )
            LOGGER.debug(
                "Correcting misread list item number %d -> %d: %r",
                num, correct, stripped[:60],
            )
            lines[i] = new_line
            expected_next = correct + 1
            changed = True

        return "\n".join(lines) if changed else markdown

    def _remove_headers_footers(self, markdown: str) -> str:
        """
        Detect lines that repeat on ≥ threshold fraction of pages and
        remove them.  Typical culprits: company names, document titles,
        ``Page N of M``, confidentiality banners.
        """
        pages = self._split_into_pages(markdown)
        if len(pages) < 3:
            # Not enough pages to reliably detect headers/footers
            return markdown

        # Collect the first 3 and last 3 non-blank lines of each page
        # (headers/footers live at the edges)
        candidate_counts: Counter = Counter()
        edge_size = 3

        for _page_num, _source, content in pages:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if not lines:
                continue

            top_lines = lines[:edge_size]
            bottom_lines = lines[-edge_size:]
            edge_lines = set(top_lines + bottom_lines)

            for line in edge_lines:
                normalised = self._normalise_for_hf_detection(line)
                if normalised:  # skip empty after normalisation
                    candidate_counts[normalised] += 1

        threshold = int(len(pages) * self.header_footer_threshold)
        if threshold < 2:
            threshold = 2

        repeated = {line for line, count in candidate_counts.items() if count >= threshold}
        if not repeated:
            return markdown

        LOGGER.info(
            "Detected %d header/footer patterns across %d pages",
            len(repeated), len(pages),
        )
        LOGGER.debug("Header/footer patterns: %s", repeated)

        # Remove matching lines from every page
        cleaned_parts: List[str] = []
        for page_num, source, content in pages:
            cleaned_lines: List[str] = []
            lines = content.split("\n")
            for line in lines:
                stripped = line.strip()
                normalised = self._normalise_for_hf_detection(stripped)
                if normalised and normalised in repeated:
                    continue
                cleaned_lines.append(line)

            cleaned_content = "\n".join(cleaned_lines).strip()
            cleaned_parts.append(f"<!-- page_number: {page_num} -->")
            if source:
                cleaned_parts.append(f"<!-- source: {source} -->")
            if cleaned_content:
                cleaned_parts.append(cleaned_content)

        return "\n\n".join(cleaned_parts)

    @staticmethod
    def _normalise_for_hf_detection(line: str) -> Optional[str]:
        """
        Normalise a line for header/footer detection.

        Only short, non-content lines are candidates.  Long sentences
        that happen to contain a page number are NOT treated as
        header/footer material.
        """
        if not line:
            return None

        # Skip markdown structural elements (headings, tables, lists)
        if re.match(r"^(#{1,6}\s|\||[-*+]\s|>\s|```)", line):
            return None

        # Only consider short lines as header/footer candidates
        # (long paragraphs are real content, not headers/footers)
        if len(line) > 80:
            return None

        # Normalise page-number variants so "Page 1" == "Page 2"
        normalised = re.sub(r"\bpage\s*\d+\b", "page N", line, flags=re.IGNORECASE)
        normalised = re.sub(r"\b\d+\s*/\s*\d+\b", "N/M", normalised)  # "3/25"
        normalised = re.sub(r"^\d+$", "N", normalised)  # standalone page number

        return normalised

    # ------------------------------------------------------------------
    # 2. Cross-page sentence continuity
    # ------------------------------------------------------------------

    def _fix_cross_page_sentences(self, markdown: str) -> str:
        """
        If a page ends mid-sentence and the next page starts with a
        lowercase continuation, merge them into one paragraph.

        Heuristic:
          - Previous page's last non-blank line does NOT end with a
            sentence terminator (. ? ! : ; |) or a markdown block marker.
          - Next page's first non-blank line starts with a lowercase letter
            or continues a word.
        """
        pages = self._split_into_pages(markdown)
        if len(pages) < 2:
            return markdown

        result_parts: List[str] = []
        for idx, (page_num, source, content) in enumerate(pages):
            if idx == 0:
                result_parts.append(f"<!-- page_number: {page_num} -->")
                if source:
                    result_parts.append(f"<!-- source: {source} -->")
                result_parts.append(content)
                continue

            prev_content = pages[idx - 1][2]
            prev_lines = [l for l in prev_content.split("\n") if l.strip()]
            curr_lines = [l for l in content.split("\n") if l.strip()]

            if prev_lines and curr_lines:
                last_line = prev_lines[-1].rstrip()
                first_line = curr_lines[0]

                if self._is_incomplete_sentence(last_line) and self._is_continuation(first_line):
                    LOGGER.debug(
                        "Merging cross-page sentence: p%d '...%s' + p%d '%s...'",
                        pages[idx - 1][0], last_line[-40:], page_num, first_line[:40],
                    )
                    # Append first line of current page to previous page's last line
                    # by replacing the trailing content
                    merged_line = last_line + " " + first_line.lstrip()

                    # Rebuild: swap last line of previous part, skip first line of current
                    prev_block = result_parts[-1]
                    prev_block_lines = prev_block.split("\n")
                    # Find and replace the last non-blank line
                    for i in range(len(prev_block_lines) - 1, -1, -1):
                        if prev_block_lines[i].strip():
                            prev_block_lines[i] = merged_line
                            break
                    result_parts[-1] = "\n".join(prev_block_lines)

                    # Current page without the merged first line
                    remaining = "\n".join(curr_lines[1:]).strip()
                    result_parts.append(f"<!-- page_number: {page_num} -->")
                    if source:
                        result_parts.append(f"<!-- source: {source} -->")
                    if remaining:
                        result_parts.append(remaining)
                    continue

            result_parts.append(f"<!-- page_number: {page_num} -->")
            if source:
                result_parts.append(f"<!-- source: {source} -->")
            result_parts.append(content)

        return "\n\n".join(result_parts)

    @staticmethod
    def _is_incomplete_sentence(line: str) -> bool:
        """Return True if *line* looks like it ends mid-sentence."""
        stripped = line.rstrip()
        if not stripped:
            return False
        # Skip markdown structural elements
        if stripped.startswith(("#", "|", "```", "---", "***", "- ", "* ", "> ")):
            return False
        # Skip HTML comments
        if stripped.endswith("-->"):
            return False
        # Sentence terminators
        if stripped[-1] in ".!?:;\"')":
            return False
        # Ends with a pipe (table row)
        if stripped.endswith("|"):
            return False
        return True

    @staticmethod
    def _is_continuation(line: str) -> bool:
        """Return True if *line* looks like a sentence continuation."""
        stripped = line.lstrip()
        if not stripped:
            return False
        # Starts with lowercase letter → continuation
        if stripped[0].islower():
            return True
        # Starts with a comma, semicolon, etc.
        if stripped[0] in ",;":
            return True
        return False

    # ------------------------------------------------------------------
    # 3. Style normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_style(markdown: str) -> str:
        """
        Harmonize minor formatting differences between Docling and
        Vision LLM output.
        """
        # Normalize bold syntax: **text** and __text__ → **text**
        markdown = re.sub(r"__([^_]+?)__", r"**\1**", markdown)

        # Normalize italic syntax: _text_ and *text* → *text*
        # (careful not to match already-bold ** or __)
        # Only standalone single underscores as italics
        markdown = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"*\1*", markdown)

        # Normalize bullet characters: •, ‣, ◦, ○ → -
        markdown = re.sub(r"^[•‣◦○]\s+", "- ", markdown, flags=re.MULTILINE)

        # Normalize em-dash variants used as bullets
        markdown = re.sub(r"^[—–]\s+", "- ", markdown, flags=re.MULTILINE)

        # Normalize horizontal rules: various styles → ---
        markdown = re.sub(r"^[\*\-_]{3,}\s*$", "---", markdown, flags=re.MULTILINE)

        # Ensure heading space: "#Title" → "# Title"
        markdown = re.sub(r"^(#{1,6})([^ #\n])", r"\1 \2", markdown, flags=re.MULTILINE)

        return markdown

    # ------------------------------------------------------------------
    # 4. Heading normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_main_heading(text: str) -> bool:
        """
        Determine whether *text* looks like a main / section heading.

        Returns ``True`` for:

        * **Numbered sections** – e.g. ``01 CONSULTING``, ``2 Introduction``
        * **ALL-CAPS text** – e.g. ``CONTENTS``, ``ABOUT THE REPORT``
          (at least 3 alphabetic characters, all upper-case).
        """
        text = text.strip()

        # Numbered section: "01 CONSULTING", "1 Market Overview"
        if _NUMBERED_SECTION_RE.match(text):
            return True

        # ALL-CAPS: only consider when ≥ 3 alpha characters
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) >= 3 and all(c.isupper() for c in alpha):
            return True

        return False

    @staticmethod
    def _match_heading_in_map(
        text: str,
        page_headings: List[Tuple[str, int]],
        *,
        threshold: float = 0.8,
    ) -> Optional[int]:
        """
        Find the best-matching entry in *page_headings* for *text*.

        Returns the heading level if a match is found with a
        similarity ratio >= *threshold*, otherwise ``None``.
        """
        _COLLAPSE_WS = re.compile(r"\s+")
        norm = _COLLAPSE_WS.sub(" ", text.strip()).lower()
        if not norm:
            return None

        best_ratio = 0.0
        best_level: Optional[int] = None

        for scanned_text, level in page_headings:
            scanned_norm = scanned_text.lower()

            # Exact match
            if norm == scanned_norm:
                return level

            # Containment check – only when the shorter string is ≥ 80%
            # of the longer, to avoid spurious substring matches like
            # "known heading" inside "unknown heading".
            shorter, longer = (
                (norm, scanned_norm) if len(norm) <= len(scanned_norm)
                else (scanned_norm, norm)
            )
            if (
                shorter in longer
                and len(shorter) >= 0.8 * len(longer)
            ):
                return level

            # Prefix match – handles headings that wrap across lines
            # in the PDF where the scanner may capture a slightly
            # different fragment than the markdown heading.
            MIN_PREFIX = 15  # require a meaningful prefix
            if (
                len(norm) >= MIN_PREFIX
                and len(scanned_norm) >= MIN_PREFIX
                and (norm.startswith(scanned_norm) or scanned_norm.startswith(norm))
            ):
                return level

            ratio = SequenceMatcher(None, norm, scanned_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_level = level

        return best_level if best_ratio >= threshold else None

    @staticmethod
    def _normalize_headings(
        markdown: str,
        *,
        heading_map: Optional[HeadingMap] = None,
    ) -> str:
        """
        Build a proper heading hierarchy from the merged markdown.

        **With heading map** (preferred):
        Uses font-metric data from the heading scanner to assign the
        correct level to each heading by matching the markdown text to
        the pre-scanned entries for that page.

        **Without heading map** (fallback):
        Uses text-pattern heuristics on flat pages:
        ALL-CAPS / numbered headings → H2, others → H3.

        In both cases a gap-closing pass ensures no heading jumps
        more than one level deeper than its predecessor.
        """
        lines = markdown.split("\n")

        # ---- 1. Determine current page & source for each line ------------
        current_page = 0
        current_source = ""
        line_page: Dict[int, int] = {}
        line_source: Dict[int, str] = {}
        for i, line in enumerate(lines):
            pm = _PAGE_MARKER_RE.match(line.strip())
            if pm:
                current_page = int(pm.group(1))
            sm = _SOURCE_MARKER_RE.match(line.strip())
            if sm:
                current_source = sm.group(1)
            line_page[i] = current_page
            line_source[i] = current_source

        use_map = bool(heading_map)

        # ---- 2. Build flat-page lookup (only needed without map) ---------
        page_flat_level: Dict[int, Optional[int]] = {}
        if not use_map:
            page_starts: List[int] = [
                i for i, ln in enumerate(lines)
                if _PAGE_MARKER_RE.match(ln.strip())
            ]
            if not page_starts:
                levels = {
                    len(m.group(1))
                    for m in (re.match(r"^(#{1,6})\s+", l) for l in lines)
                    if m
                }
                flat_level: Optional[int] = (
                    next(iter(levels)) if len(levels) == 1 else None
                )
                page_flat_level = {i: flat_level for i in range(len(lines))}
            else:
                boundaries = page_starts + [len(lines)]
                for idx in range(len(page_starts)):
                    start = boundaries[idx]
                    end = boundaries[idx + 1]
                    levels = {
                        len(m.group(1))
                        for m in (
                            re.match(r"^(#{1,6})\s+", lines[j])
                            for j in range(start, end)
                        )
                        if m
                    }
                    fl: Optional[int] = (
                        next(iter(levels)) if len(levels) == 1 else None
                    )
                    for j in range(start, end):
                        page_flat_level[j] = fl

        # ---- 3. Assign heading levels ------------------------------------
        result: List[str] = []
        last_level = 0

        for i, line in enumerate(lines):
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                hashes, text = m.group(1), m.group(2)
                level = len(hashes)

                if use_map:
                    # --- heading-map mode ---
                    page = line_page[i]
                    page_entries = heading_map.get(page, [])  # type: ignore[union-attr]
                    mapped_level = PostMergeProcessor._match_heading_in_map(
                        text, page_entries
                    )
                    if mapped_level is not None:
                        level = mapped_level
                    elif page_entries and line_source.get(i) != "vision_llm":
                        # Unmatched heading on a Docling page that has
                        # scanner data → demote to H4 (lowest sub-heading).
                        # Vision LLM pages are excluded because visual
                        # headings may not appear in the font-metric scan.
                        level = 4
                else:
                    # --- heuristic mode (fallback) ---
                    fl = page_flat_level.get(i)
                    if fl is not None:
                        if fl == 1:
                            level = 2
                        elif PostMergeProcessor._is_main_heading(text):
                            level = 2
                        else:
                            level = 3

                # Close gaps > 1 (e.g. ## → #### becomes ## → ###)
                if last_level > 0 and level > last_level + 1:
                    level = last_level + 1

                last_level = level
                result.append(f"{'#' * level} {text}")
            else:
                result.append(line)

        return "\n".join(result)

    # ------------------------------------------------------------------
    # 4b. Metric-heading conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_metric_headings(markdown: str) -> str:
        """
        Convert headings whose text is a standalone metric value into
        bullet points merged with the following non-empty line.

        Example::

            ### 84%

            Renewable Energy (% of total consumption)

        becomes::

            - **84%** — Renewable Energy (% of total consumption)

        This prevents metric values from polluting the heading hierarchy
        and improves retrieval quality by keeping the metric and its
        description together as a single chunk.
        """
        lines = markdown.split("\n")
        result: List[str] = []
        skip_next_blank = False

        i = 0
        while i < len(lines):
            line = lines[i]

            m = _METRIC_HEADING_RE.match(line)
            if m:
                metric_text = m.group(2).strip()

                # Look ahead for the next non-blank, non-comment line
                # to merge as context
                context_line = ""
                j = i + 1
                while j < len(lines):
                    candidate = lines[j].strip()
                    if not candidate:
                        j += 1
                        continue
                    # Skip HTML comments (page/source markers)
                    if candidate.startswith("<!--"):
                        break
                    context_line = candidate
                    break

                if context_line:
                    result.append(f"- **{metric_text}** — {context_line}")
                    # Skip blank lines between the heading and the
                    # context line, plus the context line itself
                    i += 1
                    while i < j:
                        i += 1
                    i = j + 1  # skip past the consumed context line
                else:
                    # No context found – just convert to a bullet
                    result.append(f"- **{metric_text}**")
                    i += 1
                continue

            result.append(line)
            i += 1

        return "\n".join(result)

    # ------------------------------------------------------------------
    # 5. Heading deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_boundary_headings(markdown: str) -> str:
        """
        Remove duplicate heading lines that appear at page boundaries.

        Specifically: if the **last heading** on page N has the same text
        as the **first heading** on page N+1, the duplicate on page N+1
        is removed.  This handles the common case where both Docling and
        Vision LLM emit the same section title at a page break.

        The comparison is case-insensitive and ignores heading level.
        """
        pages = PostMergeProcessor._split_into_pages(markdown)
        if len(pages) < 2:
            return markdown

        def _last_heading(content: str) -> Optional[str]:
            """Return the text of the last heading in *content*, or None."""
            for line in reversed(content.split("\n")):
                m = re.match(r"^#{1,6}\s+(.+)$", line.strip())
                if m:
                    return m.group(1).strip().lower()
                if line.strip():  # non-blank non-heading → stop
                    return None
            return None

        def _strip_first_heading_if(content: str, heading_text: str) -> str:
            """Remove the first heading in *content* if its text matches."""
            lines = content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                m = re.match(r"^#{1,6}\s+(.+)$", stripped)
                if m and m.group(1).strip().lower() == heading_text:
                    LOGGER.debug("Removed duplicate boundary heading: %s", stripped)
                    lines.pop(i)
                    return "\n".join(lines)
                break  # first non-blank line is not a matching heading
            return content

        result_parts: List[str] = []
        for idx, (page_num, source, content) in enumerate(pages):
            if idx > 0:
                prev_last = _last_heading(pages[idx - 1][2])
                if prev_last is not None:
                    content = _strip_first_heading_if(content, prev_last)
            result_parts.append(f"<!-- page_number: {page_num} -->")
            if source:
                result_parts.append(f"<!-- source: {source} -->")
            if content.strip():
                result_parts.append(content.strip())

        return "\n\n".join(result_parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_pages(markdown: str) -> List[Tuple[int, str, str]]:
        """
        Split merged markdown back into ``[(page_num, source, content), ...]``
        using ``<!-- page_number: N -->`` markers.

        *source* is the extraction method (e.g. ``vision_llm``, ``docling``)
        parsed from any ``<!-- source: ... -->`` comment that immediately
        follows the page marker.  If absent, *source* is an empty string.
        """
        parts = _PAGE_MARKER_RE.split(markdown)
        pages: List[Tuple[int, str, str]] = []

        # parts alternates: [pre-content, page_num, content, page_num, content, ...]
        i = 1  # skip any content before the first page marker
        while i < len(parts) - 1:
            page_num = int(parts[i])
            content = parts[i + 1].strip()

            # Extract and strip the source marker from the content
            source = ""
            source_match = _SOURCE_MARKER_RE.match(content)
            if source_match:
                source = source_match.group(1)
                content = content[source_match.end():].strip()

            pages.append((page_num, source, content))
            i += 2

        return pages
