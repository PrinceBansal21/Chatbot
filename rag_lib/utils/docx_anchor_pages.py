"""Anchor-injection page marker strategy for DOCX/DOC files.

Most accurate page-marker strategy available without Microsoft Word:

1. Open the source DOCX with ``python-docx`` and prepend a unique
   token (``⟦PMK<n>⟧``) to every non-empty paragraph (body, tables,
   nested cells).
2. Save to a temp DOCX.
3. Render that anchored DOCX → PDF via LibreOffice headless and
   extract per-page text with ``pdfplumber``. Tokens are short, plain
   ASCII-ish strings that survive the render unmodified, so we can
   build a deterministic ``anchor_id → page_number`` map.
4. Run Docling on the *same* anchored DOCX. The tokens travel into
   the markdown verbatim because Docling treats them as ordinary
   paragraph text.
5. Walk the markdown, splitting on token positions; each segment's
   page number comes straight from the map. Inject
   ``<!-- page_number: N -->`` at every transition and strip the
   tokens from the final output.

This avoids the failure mode of the older fingerprint approach
(``docx_layout_pages``), which mis-counts pages when Docling fuses
multiple source paragraphs into a single markdown block.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Anchor token format. Unicode mathematical brackets are virtually
# unheard-of in real prose, so collisions and accidental whitespace
# splitting in PDF/markdown are not a concern.
_ANCHOR_RE = re.compile(r"⟦PMK(\d+)⟧")
_ANCHOR_FMT = "⟦PMK{n}⟧"


def _find_libreoffice() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _inject_anchors_into_docx(src_path: str, dst_path: str) -> int:
    """Stamp every non-empty paragraph in ``src_path`` with a unique
    anchor token and save to ``dst_path``.

    Returns the number of anchors inserted (0 if python-docx is
    unavailable or the document has no text).
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except ImportError:
        LOGGER.info("python-docx not installed; anchor strategy skipped")
        return 0

    doc = Document(src_path)
    counter = [0]

    def _stamp(para) -> None:
        text = "".join((r.text or "") for r in para.runs)
        if not text.strip():
            return
        counter[0] += 1
        anchor = _ANCHOR_FMT.format(n=counter[0])

        # Build <w:r><w:rPr><w:sz w:val="4"/><w:szCs w:val="4"/></w:rPr>
        # <w:t xml:space="preserve">⟦PMK<n>⟧</w:t></w:r>
        # Font size 4 half-points = 2pt → keeps original page count
        # while remaining 100% extractable by pdfplumber/Docling.
        from docx.oxml import OxmlElement

        new_r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        for tag in ("w:sz", "w:szCs"):
            sz = OxmlElement(tag)
            sz.set(qn("w:val"), "4")
            rpr.append(sz)
        new_r.append(rpr)

        new_t = OxmlElement("w:t")
        new_t.text = anchor
        new_t.set(qn("xml:space"), "preserve")
        new_r.append(new_t)

        # Insert as the first non-pPr child of <w:p>.
        inserted = False
        for child in list(para._element):
            tag = child.tag.split("}", 1)[-1]
            if tag != "pPr":
                child.addprevious(new_r)
                inserted = True
                break
        if not inserted:
            para._element.append(new_r)

    def _walk(container) -> None:
        for para in container.paragraphs:
            _stamp(para)
        for table in container.tables:
            for row in table.rows:
                for cell in row.cells:
                    _walk(cell)

    _walk(doc)

    # Stamp paragraphs in headers/footers too — but mark them so we
    # can ignore those anchors (headers/footers repeat on every page
    # and would otherwise pollute the map). We simply do NOT stamp
    # them: only body content participates in the page map.

    doc.save(dst_path)
    return counter[0]


def _convert_to_pdf(soffice: str, docx_path: str, out_dir: str) -> Optional[str]:
    cmd = [
        soffice,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", out_dir,
        docx_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOGGER.warning("LibreOffice DOCX→PDF failed: %s", exc)
        return None
    if result.returncode != 0:
        LOGGER.warning(
            "LibreOffice DOCX→PDF returned %d: %s",
            result.returncode, (result.stderr or "").strip(),
        )
        return None
    pdf = os.path.join(out_dir, Path(docx_path).stem + ".pdf")
    if not os.path.isfile(pdf):
        LOGGER.warning("Expected PDF not found after conversion: %s", pdf)
        return None
    return pdf


def _build_anchor_page_map(pdf_path: str) -> Dict[int, int]:
    """Return ``{anchor_id: 1_based_page_number}`` from a rendered PDF."""
    try:
        import pdfplumber
    except ImportError:
        LOGGER.info("pdfplumber not installed; anchor strategy skipped")
        return {}

    mapping: Dict[int, int] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                for m in _ANCHOR_RE.finditer(text):
                    idx = int(m.group(1))
                    # First occurrence wins — handles the rare case
                    # where a paragraph spans two pages and the
                    # anchor appears in both.
                    mapping.setdefault(idx, page_no)
    except Exception as exc:
        LOGGER.warning("Failed to extract anchors from PDF %s: %s", pdf_path, exc)
        return {}
    return mapping


def inject_anchor_aware_page_markers(
    docx_path: str,
    docling_convert_fn: Callable[[str], str],
) -> Optional[str]:
    """Anchor-driven page-marker injection for a DOCX.

    Args:
        docx_path: Source DOCX file path.
        docling_convert_fn: Callable that takes a DOCX path and returns
            the Docling markdown (no page markers necessary).

    Returns:
        Markdown with ``<!-- page_number: N -->`` markers, or ``None``
        if any prerequisite is missing or any step fails.
    """
    soffice = _find_libreoffice()
    if not soffice:
        LOGGER.info("LibreOffice not available; anchor strategy skipped")
        return None

    workdir = tempfile.mkdtemp(prefix="anchorpages_")
    try:
        anchored_docx = os.path.join(workdir, "anchored.docx")
        n_anchors = _inject_anchors_into_docx(docx_path, anchored_docx)
        if n_anchors == 0:
            LOGGER.info(
                "No anchors injected for %s; anchor strategy skipped",
                docx_path,
            )
            return None
        LOGGER.info("Anchor strategy: inserted %d anchors", n_anchors)

        pdf_dir = os.path.join(workdir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = _convert_to_pdf(soffice, anchored_docx, pdf_dir)
        if not pdf_path:
            return None

        anchor_to_page = _build_anchor_page_map(pdf_path)
        if not anchor_to_page:
            LOGGER.warning(
                "Anchor strategy: no anchors found in PDF for %s", docx_path,
            )
            return None
        coverage = len(anchor_to_page) / n_anchors
        LOGGER.info(
            "Anchor strategy: mapped %d/%d anchors (%.1f%%) across %d page(s)",
            len(anchor_to_page), n_anchors, coverage * 100,
            max(anchor_to_page.values()),
        )

        try:
            anchored_md = docling_convert_fn(anchored_docx)
        except Exception as exc:
            LOGGER.warning(
                "Docling conversion of anchored DOCX failed for %s: %s",
                docx_path, exc,
            )
            return None
        if not anchored_md or "⟦PMK" not in anchored_md:
            LOGGER.warning(
                "Anchor strategy: anchors did not survive Docling for %s",
                docx_path,
            )
            return None

        # Find every anchor occurrence in the markdown (in order).
        positions = [
            (m.start(), m.end(), int(m.group(1)))
            for m in _ANCHOR_RE.finditer(anchored_md)
        ]

        # Compose output: each anchor opens a segment that lasts until
        # the next anchor (or EOF). Insert page markers at transitions.
        out_parts = []
        last_page: Optional[int] = None

        # Preserve any preamble before the first anchor on the first page.
        preamble = anchored_md[: positions[0][0]].strip()
        if preamble:
            first_page = anchor_to_page.get(positions[0][2], 1)
            out_parts.append(f"<!-- page_number: {first_page} -->")
            out_parts.append(preamble)
            last_page = first_page

        for i, (start, end, aid) in enumerate(positions):
            page = anchor_to_page.get(aid, last_page if last_page else 1)
            seg_end = positions[i + 1][0] if i + 1 < len(positions) else len(anchored_md)
            # Drop the anchor token itself, keep everything after it.
            segment = anchored_md[end:seg_end]
            segment = _ANCHOR_RE.sub("", segment).strip()
            if not segment:
                continue
            if page != last_page:
                out_parts.append(f"<!-- page_number: {page} -->")
                last_page = page
            out_parts.append(segment)

        result_md = "\n\n".join(out_parts)
        # Defensive: strip any leftover anchor tokens.
        result_md = _ANCHOR_RE.sub("", result_md)

        LOGGER.info(
            "Anchor-aware page markers injected for %s: %d segments, %d unique pages",
            docx_path, len(out_parts),
            len({anchor_to_page[a] for a in anchor_to_page}),
        )
        return result_md
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
