"""Unified single-pass PDF pre-scan.

Replaces three separate pre-scan passes (heading_scanner, pdf_orientation,
page analysis) with **one** ``pdfplumber.open()`` + **one** ``PdfReader()``
that extracts everything the parallel hybrid adapter needs:

- Heading structure (font-size-based heading hierarchy)
- Per-page rotation metadata and text-orientation analysis
- Per-page routing decision (Docling vs Vision LLM)
- Per-page infographic flags and image/text coverage

For a 100-page PDF this reduces ~302 PDF opens to **2**.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
    _PYPDF = True
except ImportError:
    _PYPDF = False

try:
    import pdfplumber as _pdfplumber
    _PLUMBER = True
except ImportError:
    _PLUMBER = False

# Re-use heading-scan helpers ---------------------------------------------------
from rag_lib.adapters.markdown.parallel_hybrid_pdf.heading_scanner import (
    _normalise_text,
    _is_metric_value,
)

# Type aliases
HeadingMap = Dict[int, List[Tuple[str, int]]]


@dataclass
class PageInfo:
    """Analysis results for a single PDF page."""
    page_num: int               # 1-based
    method: str                 # 'vision' | 'docling'
    is_infographic: bool
    text_coverage: float
    image_coverage: float
    has_tables: bool
    rotation_needed: int        # degrees (0, 90, 180, 270)
    rotation_method: str        # 'pdf_metadata' | 'text_analysis' | 'none_needed'


@dataclass
class PreScanResult:
    """Complete pre-scan output for the entire PDF."""
    heading_map: HeadingMap = field(default_factory=dict)
    pages: Dict[int, PageInfo] = field(default_factory=dict)
    total_pages: int = 0
    needs_rotation_correction: bool = False

    # Convenience views --------------------------------------------------------
    @property
    def docling_pages(self) -> List[int]:
        return sorted(p for p, info in self.pages.items() if info.method == 'docling')

    @property
    def vision_pages(self) -> List[int]:
        return sorted(p for p, info in self.pages.items() if info.method == 'vision')

    @property
    def infographic_page_list(self) -> List[int]:
        return sorted(p for p, info in self.pages.items() if info.is_infographic)

    @property
    def rotation_map(self) -> Dict[int, int]:
        return {p: info.rotation_needed for p, info in self.pages.items()
                if info.rotation_needed != 0}

    @property
    def visual_gap_pages(self) -> List[int]:
        """Docling pages with high image coverage needing visual gap detection."""
        return sorted(
            p for p, info in self.pages.items()
            if info.method == 'docling' and info.image_coverage >= 0.3
        )


def unified_prescan(
    pdf_path: str,
    *,
    text_threshold: float = 0.3,
    always_vision_first_page: bool = True,
    infographic_pages: Optional[Set[int]] = None,
    table_pages: Optional[Set[int]] = None,
) -> PreScanResult:
    """Run a single-pass pre-scan of the PDF.

    Opens ``pdfplumber`` once and ``PdfReader`` once to gather:
    heading structure, rotation metadata, text/image/table analysis, and
    per-page routing decisions.

    Args:
        pdf_path: Path to the PDF file.
        text_threshold: Minimum text coverage to route to Docling.
        always_vision_first_page: Route page 1 to Vision LLM.
        infographic_pages: Force these pages (1-based) to Vision LLM.
        table_pages: Force these pages (1-based) to Docling.

    Returns:
        :class:`PreScanResult` with all per-page information.
    """
    infographic_pages = infographic_pages or set()
    table_pages = table_pages or set()
    result = PreScanResult()

    # ------------------------------------------------------------------
    # Phase A: pypdf — rotation metadata for ALL pages (one open)
    # ------------------------------------------------------------------
    rotation_metadata: Dict[int, int] = {}  # page_num (1-based) → degrees
    if _PYPDF:
        try:
            reader = PdfReader(pdf_path)
            result.total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                rot = int(page.get("/Rotate") or 0) % 360
                if rot != 0:
                    rotation_metadata[idx + 1] = rot
        except Exception as exc:
            LOGGER.debug("pypdf rotation scan failed: %s", exc)

    # ------------------------------------------------------------------
    # Phase B: pdfplumber — heading scan + routing + text orientation
    # (one open, iterate all pages)
    # ------------------------------------------------------------------
    if not _PLUMBER:
        LOGGER.warning("pdfplumber not available — pre-scan limited to metadata")
        # Fallback: produce minimal results
        for pn in range(1, result.total_pages + 1):
            rot = rotation_metadata.get(pn, 0)
            result.pages[pn] = PageInfo(
                page_num=pn,
                method='docling',
                is_infographic=pn in infographic_pages,
                text_coverage=0.5, image_coverage=0.0, has_tables=False,
                rotation_needed=rot,
                rotation_method='pdf_metadata' if rot else 'none_needed',
            )
        result.needs_rotation_correction = bool(rotation_metadata)
        return result

    try:
        pdf = _pdfplumber.open(pdf_path)
    except Exception as exc:
        LOGGER.warning("Could not open %s for pre-scan: %s", pdf_path, exc)
        return result

    result.total_pages = len(pdf.pages)

    # --- Heading scan accumulators ---
    LINE_Y_TOLERANCE = 3.0
    LineInfo = Tuple[int, float, bool, str]
    all_lines: List[LineInfo] = []

    for page_idx in range(len(pdf.pages)):
        page_num = page_idx + 1
        page = pdf.pages[page_idx]

        # ---- Heading data extraction (chars) ----
        try:
            chars = page.chars or []
        except Exception:
            chars = []

        if chars:
            _collect_heading_lines(chars, page_num, LINE_Y_TOLERANCE, all_lines)

        # ---- Page routing analysis ----
        # Forced pages — skip expensive analysis
        if page_num in infographic_pages:
            rot = rotation_metadata.get(page_num, 0)
            result.pages[page_num] = PageInfo(
                page_num=page_num, method='vision', is_infographic=True,
                text_coverage=0.0, image_coverage=1.0, has_tables=False,
                rotation_needed=rot,
                rotation_method='pdf_metadata' if rot else 'none_needed',
            )
            continue
        if page_num in table_pages:
            rot = rotation_metadata.get(page_num, 0)
            result.pages[page_num] = PageInfo(
                page_num=page_num, method='docling', is_infographic=False,
                text_coverage=0.5, image_coverage=0.0, has_tables=True,
                rotation_needed=rot,
                rotation_method='pdf_metadata' if rot else 'none_needed',
            )
            continue

        # Page metrics
        page_area = page.width * page.height
        if page_area <= 0:
            result.pages[page_num] = PageInfo(
                page_num=page_num, method='docling', is_infographic=False,
                text_coverage=0.0, image_coverage=0.0, has_tables=False,
                rotation_needed=0, rotation_method='none_needed',
            )
            continue

        words = page.extract_words() or []
        text_area = sum(
            (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
        )
        images = page.images or []
        image_area = sum(
            (img["x1"] - img["x0"]) * (img["bottom"] - img["top"]) for img in images
        )
        has_tables = False
        try:
            tables = page.find_tables() or []
            has_tables = len(tables) > 0
        except Exception:
            pass

        text_coverage = text_area / page_area
        image_coverage = image_area / page_area

        # Routing decision
        is_infographic = image_coverage > 0.3 and text_coverage < text_threshold

        if page_num == 1 and always_vision_first_page:
            method = 'vision'
            is_infographic = True
        elif has_tables:
            method = 'docling'
        elif is_infographic:
            method = 'vision'
        elif text_coverage < 0.1:
            method = 'vision'
            is_infographic = text_coverage < 0.1 and image_coverage > 0.1
        else:
            method = 'docling'

        # Text-orientation check (reuse the already-extracted words)
        rot = rotation_metadata.get(page_num, 0)
        rot_method = 'pdf_metadata' if rot else 'none_needed'
        if rot == 0 and words:
            text_rot = _detect_text_rotation(words)
            if text_rot != 0:
                rot = text_rot
                rot_method = 'text_analysis'

        result.pages[page_num] = PageInfo(
            page_num=page_num, method=method, is_infographic=is_infographic,
            text_coverage=text_coverage, image_coverage=image_coverage,
            has_tables=has_tables,
            rotation_needed=rot, rotation_method=rot_method,
        )

    pdf.close()
    result.needs_rotation_correction = any(
        info.rotation_needed != 0 for info in result.pages.values()
    )

    # --- Build heading map from accumulated lines ---
    result.heading_map = _build_heading_map(all_lines)

    # Log summary
    docling_count = len(result.docling_pages)
    vision_count = len(result.vision_pages)
    infographic_count = len(result.infographic_page_list)
    LOGGER.info(
        "Unified pre-scan complete: %d pages, %d→docling, %d→vision, "
        "%d infographic, %d need rotation, %d headings detected",
        result.total_pages, docling_count, vision_count, infographic_count,
        len(result.rotation_map),
        sum(len(v) for v in result.heading_map.values()),
    )
    return result


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _detect_text_rotation(words: list, sample_lines: int = 40) -> int:
    """Detect text orientation from pdfplumber words (already extracted)."""
    horizontal = 0
    vertical = 0
    for word in words[:sample_lines]:
        width = word["x1"] - word["x0"]
        height = word["bottom"] - word["top"]
        if width > height * 1.5:
            horizontal += 1
        elif height > width * 1.5:
            vertical += 1
    if vertical > horizontal * 2:
        return 90
    return 0


def _collect_heading_lines(
    chars: list,
    page_num: int,
    line_y_tolerance: float,
    out: List,
) -> None:
    """Extract heading-candidate lines from page characters, appending to *out*."""
    sorted_chars = sorted(
        (c for c in chars if c.get("text", "").strip()),
        key=lambda c: (round(c["top"] / line_y_tolerance), c["x0"]),
    )

    raw_lines: List[List[dict]] = []
    current_line: List[dict] = []
    current_y: Optional[float] = None

    for ch in sorted_chars:
        ch_y = ch["top"]
        if current_y is None or abs(ch_y - current_y) <= line_y_tolerance:
            current_line.append(ch)
            current_y = ch_y if current_y is None else current_y
        else:
            if current_line:
                raw_lines.append(current_line)
            current_line = [ch]
            current_y = ch_y
    if current_line:
        raw_lines.append(current_line)

    current_group: Optional[Tuple[float, bool, float, List[str]]] = None

    for line_chars in raw_lines:
        if not line_chars:
            continue
        line_text = "".join(c.get("text", "") for c in line_chars).strip()
        if not line_text:
            continue

        size_counts: Counter = Counter()
        bold_chars = 0
        for c in line_chars:
            sz = round(c.get("size", 0), 1)
            size_counts[sz] += 1
            if "bold" in (c.get("fontname") or "").lower():
                bold_chars += 1

        font_size = size_counts.most_common(1)[0][0] if size_counts else 0
        is_bold = bold_chars > len(line_chars) * 0.5
        line_y0 = line_chars[0]["top"]

        can_merge = (
            current_group is not None
            and current_group[0] == font_size
            and current_group[1] == is_bold
            and (line_y0 - current_group[2]) > font_size * 0.5
        )

        if can_merge:
            current_group[3].append(line_text)
            current_group = (current_group[0], current_group[1], line_y0, current_group[3])
        else:
            if current_group is not None:
                out.append((page_num, current_group[0], current_group[1],
                            " ".join(current_group[3])))
            current_group = (font_size, is_bold, line_y0, [line_text])

    if current_group is not None:
        out.append((page_num, current_group[0], current_group[1],
                    " ".join(current_group[3])))


def _build_heading_map(all_lines: list) -> HeadingMap:
    """Build the heading map from collected line info — identical logic to heading_scanner."""
    if not all_lines:
        return {}

    # Body text size
    size_chars: Counter = Counter()
    for _, size, _, text in all_lines:
        size_chars[size] += len(text)
    body_size = size_chars.most_common(1)[0][0]

    # Heading font sizes
    MIN_DELTA = 1.5
    raw_heading_sizes = sorted(
        {s for s in size_chars if s >= body_size + MIN_DELTA}, reverse=True,
    )

    CLUSTER_GAP = 3.0
    heading_sizes: List[float] = []
    size_cluster_map: Dict[float, float] = {}
    for size in raw_heading_sizes:
        if not heading_sizes or (heading_sizes[-1] - size) >= CLUSTER_GAP:
            heading_sizes.append(size)
        size_cluster_map[size] = heading_sizes[-1]

    MAX_HEADING_LEVELS = 2
    heading_sizes = heading_sizes[:MAX_HEADING_LEVELS]

    cluster_to_level: Dict[float, int] = {}
    for idx, size in enumerate(heading_sizes):
        cluster_to_level[size] = 2 + idx

    size_to_level: Dict[float, int] = {}
    for raw_size, cluster_rep in size_cluster_map.items():
        if cluster_rep in cluster_to_level:
            size_to_level[raw_size] = cluster_to_level[cluster_rep]

    bold_body_level = max(cluster_to_level.values(), default=3)
    bold_body_level = min(bold_body_level, 6)

    MAX_HEADING_LEN = 80
    MIN_HEADING_LEN = 3
    heading_map: HeadingMap = {}

    for page_num, font_size, is_bold, text in all_lines:
        normalised = _normalise_text(text)
        if not normalised or len(normalised) > MAX_HEADING_LEN or len(normalised) < MIN_HEADING_LEN:
            continue
        if _is_metric_value(normalised):
            continue

        level: Optional[int] = None
        if font_size in size_to_level:
            level = size_to_level[font_size]
        elif is_bold and font_size >= body_size - 0.5:
            level = bold_body_level

        if level is not None:
            heading_map.setdefault(page_num, []).append((normalised, level))

    return heading_map
