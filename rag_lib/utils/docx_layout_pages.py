"""Layout-aware DOCX page marker injection.

Given a DOCX (or DOCX-converted-from-DOC) file and the markdown produced by
Docling's DOCX adapter, inject ``<!-- page_number: N -->`` markers at the
positions where pages actually break in the rendered document.

Approach
--------
1. Convert the DOCX to PDF using LibreOffice headless. The PDF is used **only**
   as a layout oracle — it is discarded after extracting per-page text.
2. Extract plain text per page from the PDF using ``pdfplumber``.
3. Walk through the markdown paragraph-by-paragraph. For each markdown block,
   compute a normalized text fingerprint and find which PDF page it belongs
   to via substring search, advancing a cursor so order is preserved.
4. Inject page markers wherever the page index changes between consecutive
   markdown blocks.

Why this is preferred over DOCX → PDF → pipeline
-------------------------------------------------
We keep the **native** Docling DOCX adapter output (which has clean headings,
tables, lists, embedded image semantics) and *only* borrow the page boundaries
from the PDF render. Structure quality is never degraded by going through a
PDF intermediate.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Minimum length of normalized markdown block text required to attempt
# fingerprint matching. Shorter blocks are assigned to the current page
# without searching (avoids false matches on common short strings).
_MIN_FINGERPRINT_LEN = 12

# Length of the prefix used as the fingerprint for searching in PDF text.
_FINGERPRINT_LEN = 60


def _find_libreoffice() -> Optional[str]:
    """Return path to soffice/libreoffice or ``None`` if not installed."""
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _convert_docx_to_pdf(docx_path: str) -> Optional[str]:
    """Convert a DOCX file to PDF using LibreOffice. Returns PDF path or None on failure.

    Caller is responsible for cleaning up the parent temp directory.
    """
    soffice = _find_libreoffice()
    if not soffice:
        LOGGER.warning("LibreOffice not available; cannot use layout-aware page markers")
        return None

    output_dir = tempfile.mkdtemp(prefix="layoutpages_")
    cmd = [
        soffice,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        docx_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOGGER.warning("LibreOffice DOCX→PDF conversion failed: %s", exc)
        shutil.rmtree(output_dir, ignore_errors=True)
        return None

    if result.returncode != 0:
        LOGGER.warning(
            "LibreOffice DOCX→PDF conversion returned %d: %s",
            result.returncode, (result.stderr or "").strip(),
        )
        shutil.rmtree(output_dir, ignore_errors=True)
        return None

    pdf_name = Path(docx_path).stem + ".pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    if not os.path.isfile(pdf_path):
        LOGGER.warning("Expected PDF not found after conversion: %s", pdf_path)
        shutil.rmtree(output_dir, ignore_errors=True)
        return None

    return pdf_path


def _extract_pdf_pages(pdf_path: str) -> List[str]:
    """Return a list of normalized text strings, one per PDF page."""
    try:
        import pdfplumber
    except ImportError:
        LOGGER.warning("pdfplumber not installed; cannot extract page text")
        return []

    pages: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(_normalize(text))
    except Exception as exc:
        LOGGER.warning("Failed to extract text from PDF %s: %s", pdf_path, exc)
        return []

    return pages


def _strip_headers_footers(pages: List[str]) -> List[str]:
    """Remove repeated short lines that appear at the top/bottom of most pages.

    Heuristic: collect the first and last non-empty line of each page; any line
    shorter than 80 chars that appears on more than half the pages is stripped.
    """
    if len(pages) < 3:
        return pages

    raw_lines: List[List[str]] = [
        [ln for ln in p.split("\n") if ln.strip()] for p in pages
    ]
    candidates: dict = {}
    for lines in raw_lines:
        if not lines:
            continue
        for ln in (lines[0], lines[-1]):
            if 0 < len(ln) < 80:
                candidates[ln] = candidates.get(ln, 0) + 1

    threshold = max(2, len(pages) // 2)
    repeated = {ln for ln, count in candidates.items() if count >= threshold}
    if not repeated:
        return pages

    cleaned: List[str] = []
    for lines in raw_lines:
        cleaned.append("\n".join(ln for ln in lines if ln not in repeated))
    return cleaned


_MD_DECORATION_RE = re.compile(
    r"(!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|<!--[^>]*-->|[#>*_`~|\\])"
)


def _normalize(text: str) -> str:
    """Normalize text for fingerprint comparison.

    - NFKC unicode normalization
    - Lowercase
    - Collapse all whitespace runs to a single space
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _block_fingerprint(block: str) -> str:
    """Extract a normalized fingerprint from a markdown block for searching."""
    # Strip markdown decorations and link/image targets, keep only visible text
    cleaned = _MD_DECORATION_RE.sub(" ", block)
    # Collapse pipes/dashes from tables
    cleaned = re.sub(r"[-=]{2,}", " ", cleaned)
    normalized = _normalize(cleaned)
    return normalized[:_FINGERPRINT_LEN]


def _find_page_for_block(
    fingerprint: str,
    pages: List[str],
    cursor: List[int],
) -> Optional[int]:
    """Return the 1-based page number for a fingerprint, advancing the cursor.

    ``cursor`` is a 2-element list: [current_page_index_0_based, char_offset_in_page].
    The function searches forward starting at the cursor and only moves forward.
    Returns ``None`` if no match is found anywhere from the cursor onward.
    """
    if not fingerprint or len(fingerprint) < _MIN_FINGERPRINT_LEN:
        return None

    page_idx, offset = cursor

    # Search current page from offset onward
    for i in range(page_idx, len(pages)):
        haystack = pages[i]
        start = offset if i == page_idx else 0
        pos = haystack.find(fingerprint, start)
        if pos >= 0:
            cursor[0] = i
            cursor[1] = pos + len(fingerprint)
            return i + 1  # 1-based page number

    return None


def inject_layout_aware_page_markers(
    markdown: str,
    docx_path: str,
) -> Optional[str]:
    """Inject page markers into ``markdown`` based on the layout of ``docx_path``.

    Returns the augmented markdown on success, or ``None`` if any step fails
    (missing LibreOffice, conversion error, no extractable PDF text, etc.) so
    the caller can fall back to a different strategy.
    """
    if not markdown or not markdown.strip():
        return markdown

    pdf_path = _convert_docx_to_pdf(docx_path)
    if not pdf_path:
        return None
    pdf_dir = str(Path(pdf_path).parent)

    try:
        pages = _extract_pdf_pages(pdf_path)
    finally:
        shutil.rmtree(pdf_dir, ignore_errors=True)

    if not pages:
        LOGGER.info("No PDF pages extracted for %s; layout-aware markers skipped", docx_path)
        return None

    pages = _strip_headers_footers(pages)
    num_pages = len(pages)

    # Single-page document: short-circuit.
    if num_pages == 1:
        return f"<!-- page_number: 1 -->\n\n{markdown.strip()}"

    blocks = [b for b in re.split(r"\n{2,}", markdown.strip()) if b.strip()]
    if not blocks:
        return markdown

    cursor = [0, 0]  # [page_index_0_based, char_offset_in_page]
    block_pages: List[int] = []
    current_page = 1

    for block in blocks:
        fp = _block_fingerprint(block)
        matched = _find_page_for_block(fp, pages, cursor)
        if matched is not None:
            current_page = matched
        # else: keep current_page (carry forward)
        block_pages.append(current_page)

    # Build output, injecting markers at page transitions.
    output_parts: List[str] = []
    last_page: Optional[int] = None
    matched_count = 0
    for block, page in zip(blocks, block_pages):
        if page != last_page:
            output_parts.append(f"<!-- page_number: {page} -->")
            last_page = page
            matched_count += 1
        output_parts.append(block)

    LOGGER.info(
        "Layout-aware page markers injected for %s: %d pages, %d blocks, "
        "%d page transitions",
        docx_path, num_pages, len(blocks), matched_count,
    )

    return "\n\n".join(output_parts)
