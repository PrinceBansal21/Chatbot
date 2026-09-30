"""DOCX page-break detection from raw Open XML.

Provides a fallback for estimating page boundaries in DOCX files when
Docling's ``result.document.pages`` is empty.  This parses the
``word/document.xml`` inside the ``.docx`` ZIP archive directly, looking
for explicit page breaks, last-rendered page breaks, and section breaks.

The main entry point is :func:`count_docx_pages` which returns the
estimated number of pages, and :func:`inject_docx_page_markers` which
splits existing markdown content at approximate page boundaries based
on the detected break positions.
"""
from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Open XML namespaces
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _find_page_break_positions(docx_path: str) -> List[int]:
    """Parse ``word/document.xml`` and return 0-based paragraph indices where page breaks occur.

    Detected break types:
    * ``<w:br w:type="page"/>`` — explicit page break
    * ``<w:lastRenderedPageBreak/>`` — Word's cached page break position
    * ``<w:sectPr>`` inside ``<w:pPr>`` — section break (next paragraph starts a new page)

    Args:
        docx_path: Path to the ``.docx`` file.

    Returns:
        Sorted list of 0-based paragraph indices *after* which a page break
        occurs (i.e. the next paragraph starts on a new page).
        Returns empty list if the file cannot be parsed.
    """
    if not os.path.isfile(docx_path):
        return []

    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return []
            doc_xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, OSError) as exc:
        LOGGER.debug("Cannot open DOCX as ZIP: %s", exc)
        return []

    try:
        root = ET.fromstring(doc_xml)
    except ET.ParseError as exc:
        LOGGER.debug("Cannot parse document.xml: %s", exc)
        return []

    body = root.find(f"{_W}body")
    if body is None:
        return []

    break_after_indices: List[int] = []
    paragraphs = list(body.iter(f"{_W}p"))

    for idx, para in enumerate(paragraphs):
        has_break = False

        # Check runs for explicit page breaks and lastRenderedPageBreak
        for run in para.iter(f"{_W}r"):
            for br in run.iter(f"{_W}br"):
                br_type = br.get(f"{_W}type") or br.get("type") or ""
                if br_type == "page":
                    has_break = True
            if list(run.iter(f"{_W}lastRenderedPageBreak")):
                has_break = True

        # Section break in paragraph properties
        ppr = para.find(f"{_W}pPr")
        if ppr is not None:
            sect_pr = ppr.find(f"{_W}sectPr")
            if sect_pr is not None:
                # Check if it's a continuous section break (no new page)
                sect_type_el = sect_pr.find(f"{_W}type")
                sect_type_val = ""
                if sect_type_el is not None:
                    sect_type_val = sect_type_el.get(f"{_W}val") or sect_type_el.get("val") or ""
                if sect_type_val != "continuous":
                    has_break = True

        if has_break:
            break_after_indices.append(idx)

    return sorted(set(break_after_indices))


def count_docx_pages(docx_path: str) -> int:
    """Return the estimated number of pages in a DOCX file.

    Returns 1 if no page breaks are found (single-page document).

    Args:
        docx_path: Path to the ``.docx`` file.
    """
    breaks = _find_page_break_positions(docx_path)
    return len(breaks) + 1


def inject_docx_page_markers(markdown: str, docx_path: str) -> str:
    """Inject ``<!-- page_number: N -->`` markers into markdown based on DOCX page breaks.

    This is a **fallback** for when Docling doesn't populate ``pages``.
    It works by:
    1. Counting page breaks in the DOCX XML.
    2. Splitting the markdown into roughly equal segments per page break.
    3. Injecting page markers at paragraph boundaries closest to the split points.

    The heuristic splits on blank-line-separated blocks (markdown paragraphs)
    and distributes them proportionally across pages.

    Args:
        markdown: The full markdown content (without page markers).
        docx_path: Path to the original ``.docx`` file.

    Returns:
        Markdown with ``<!-- page_number: N -->`` markers injected.
        If only 1 page is detected, returns ``<!-- page_number: 1 -->\\n\\n`` + markdown.
    """
    breaks = _find_page_break_positions(docx_path)
    num_pages = len(breaks) + 1

    stripped = markdown.strip()
    if not stripped:
        return markdown

    # Single page — just prepend marker
    if num_pages <= 1:
        return f"<!-- page_number: 1 -->\n\n{stripped}"

    # Split markdown into paragraph-level blocks (separated by blank lines)
    blocks = re.split(r"\n{2,}", stripped)
    blocks = [b for b in blocks if b.strip()]

    if not blocks:
        return f"<!-- page_number: 1 -->\n\n{stripped}"

    # We have N page breaks → (N+1) pages.
    # Distribute blocks proportionally across pages.
    # Strategy: if we have P paragraphs in the DOCX XML and B blocks in
    # markdown, we map break positions proportionally.
    #
    # break_after_indices are 0-based paragraph indices in the XML.
    # We approximate: the fraction of the doc at which a break occurs is
    # break_idx / total_xml_paragraphs.  We apply that fraction to our
    # markdown blocks.

    # Estimate total XML paragraphs (last break index + some margin)
    max_break_idx = max(breaks) if breaks else 0
    # Assume a few more paragraphs after the last break
    estimated_xml_paragraphs = max_break_idx + max(len(blocks) // num_pages, 1)

    # Compute block indices where we should insert page breaks
    block_break_indices: List[int] = []
    for brk in breaks:
        # Proportional mapping
        frac = brk / estimated_xml_paragraphs if estimated_xml_paragraphs > 0 else 0
        block_idx = int(frac * len(blocks))
        # Clamp to valid range
        block_idx = max(1, min(block_idx, len(blocks) - 1))
        block_break_indices.append(block_idx)

    # De-duplicate and sort
    block_break_indices = sorted(set(block_break_indices))

    # Build output with page markers
    parts: List[str] = []
    current_page = 1
    break_set = set(block_break_indices)

    parts.append(f"<!-- page_number: {current_page} -->")
    for i, block in enumerate(blocks):
        if i in break_set:
            current_page += 1
            parts.append(f"\n\n<!-- page_number: {current_page} -->")
        parts.append(block)

    return "\n\n".join(parts)
