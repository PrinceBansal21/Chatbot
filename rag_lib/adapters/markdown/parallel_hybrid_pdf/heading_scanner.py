"""
Heading Scanner – pre-scan a PDF for heading structure using font metrics.

Before parallel page processing begins, this module scans the *entire*
document with PyMuPDF to extract font-size information for every text
line.  By clustering font sizes globally it produces a **heading map**
that records, for every page, which text lines are headings and at
what level they should appear.

The heading map is then consumed by :class:`PostMergeProcessor` so
that the final merged markdown reflects the true document hierarchy
rather than relying on per-page heuristics.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Type alias: { page_number: [ (heading_text, heading_level), … ] }
HeadingMap = Dict[int, List[Tuple[str, int]]]


def scan_heading_structure(pdf_path: str) -> HeadingMap:
    """
    Pre-scan a PDF and return a mapping of page → heading entries.

    Each entry is ``(normalised_text, level)`` where *level* starts at
    ``2`` (H2) for the largest heading size and increments for each
    smaller size band.

    The algorithm:

    1. Extract every text line together with its dominant font size.
    2. Determine the *body text size* – the font size that accounts for
       the most characters in the document.
    3. Collect all font sizes that are **significantly larger** than the
       body text (≥ 1.5 pt bigger).
    4. Cluster those sizes in descending order and assign heading levels
       H2, H3, H4, …
    5. **Bold text at body size** is treated as the deepest heading level
       (one level below the smallest font-size heading).

    Returns an empty dict if no headings are detected or if PyMuPDF is
    not available.
    """
    try:
        import pdfplumber
    except ImportError:
        LOGGER.warning("pdfplumber not available – heading scan skipped")
        return {}

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        LOGGER.warning("Could not open %s for heading scan: %s", pdf_path, exc)
        return {}

    # ------------------------------------------------------------------
    # 1. Collect (page, font_size, is_bold, text) for every text line
    # ------------------------------------------------------------------
    LineInfo = Tuple[int, float, bool, str]
    all_lines: List[LineInfo] = []

    for page_idx in range(len(pdf.pages)):
        page = pdf.pages[page_idx]
        try:
            chars = page.chars or []
        except Exception:
            continue

        if not chars:
            continue

        # Group characters into lines by y-coordinate similarity,
        # then merge spans that share the same font characteristics.
        # A "line" is a group of chars with similar top/bottom y.
        LINE_Y_TOLERANCE = 3.0  # pts

        # Sort by top then x0
        sorted_chars = sorted(chars, key=lambda c: (round(c["top"] / LINE_Y_TOLERANCE), c["x0"]))

        # Group chars into raw lines by y-proximity
        raw_lines: List[List[dict]] = []
        current_line: List[dict] = []
        current_y: Optional[float] = None

        for ch in sorted_chars:
            if ch.get("text", "").strip() == "":
                continue
            ch_y = ch["top"]
            if current_y is None or abs(ch_y - current_y) <= LINE_Y_TOLERANCE:
                current_line.append(ch)
                current_y = ch_y if current_y is None else current_y
            else:
                if current_line:
                    raw_lines.append(current_line)
                current_line = [ch]
                current_y = ch_y

        if current_line:
            raw_lines.append(current_line)

        # For each raw line, determine dominant font size and boldness,
        # then group consecutive lines with the same font properties
        # (merged / wrapped text).
        current_group: Optional[Tuple[float, bool, float, List[str]]] = None

        for line_chars in raw_lines:
            if not line_chars:
                continue

            line_text = "".join(c.get("text", "") for c in line_chars).strip()
            if not line_text:
                continue

            # Dominant font size = most common size in the line (weighted by char count)
            size_counts: Counter[float] = Counter()
            bold_chars = 0
            for c in line_chars:
                sz = round(c.get("size", 0), 1)
                size_counts[sz] += 1
                font_name = (c.get("fontname") or "").lower()
                if "bold" in font_name:
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
                current_group = (
                    current_group[0], current_group[1],
                    line_y0, current_group[3],
                )
            else:
                if current_group is not None:
                    all_lines.append(
                        (page_idx + 1, current_group[0], current_group[1],
                         " ".join(current_group[3]))
                    )
                current_group = (font_size, is_bold, line_y0, [line_text])

        # Flush last group
        if current_group is not None:
            all_lines.append(
                (page_idx + 1, current_group[0], current_group[1],
                 " ".join(current_group[3]))
            )

    pdf.close()

    if not all_lines:
        return {}

    # ------------------------------------------------------------------
    # 2. Determine body-text font size (weighted by character count)
    # ------------------------------------------------------------------
    size_chars: Counter[float] = Counter()
    for _, size, _, text in all_lines:
        size_chars[size] += len(text)

    body_size = size_chars.most_common(1)[0][0]
    LOGGER.debug("Heading scan – body font size: %.1f pt", body_size)

    # ------------------------------------------------------------------
    # 3. Identify heading font sizes (larger than body text)
    # ------------------------------------------------------------------
    MIN_DELTA = 1.5  # minimum pt difference to be considered a heading
    raw_heading_sizes = sorted(
        {s for s in size_chars if s >= body_size + MIN_DELTA},
        reverse=True,
    )

    # Cluster nearby sizes – sizes within CLUSTER_GAP pt of each other
    # are treated as the same heading level (maps to the larger size).
    CLUSTER_GAP = 3.0
    heading_sizes: List[float] = []
    size_cluster_map: Dict[float, float] = {}  # raw_size → cluster_representative
    for size in raw_heading_sizes:
        if not heading_sizes or (heading_sizes[-1] - size) >= CLUSTER_GAP:
            heading_sizes.append(size)
        # Map this raw size to the most recent cluster representative
        size_cluster_map[size] = heading_sizes[-1]

    # Cap at 2 distinct heading levels (H2 and H3) to keep the
    # hierarchy clean.  The scanner identifies the *structural*
    # split between major section titles (largest font) and
    # sub-headings (everything else above body size).  Further
    # nesting is left to the adapter's output or heuristic pass.
    MAX_HEADING_LEVELS = 2
    heading_sizes = heading_sizes[:MAX_HEADING_LEVELS]

    # ------------------------------------------------------------------
    # 4. Assign heading levels: largest → H2, next → H3, …
    # ------------------------------------------------------------------
    cluster_to_level: Dict[float, int] = {}
    for idx, size in enumerate(heading_sizes):
        cluster_to_level[size] = 2 + idx  # H2, H3, H4, H5

    # Build the final size → level mapping through cluster representatives
    size_to_level: Dict[float, int] = {}
    for raw_size, cluster_rep in size_cluster_map.items():
        if cluster_rep in cluster_to_level:
            size_to_level[raw_size] = cluster_to_level[cluster_rep]

    # Bold text at body-size → same level as the deepest font-size
    # heading cluster.  With a 2-level cap this means H3, which keeps
    # the heading map limited to just H2 (section titles) and H3
    # (sub-headings / bold labels).
    bold_body_level = max(cluster_to_level.values(), default=3)
    bold_body_level = min(bold_body_level, 6)

    if heading_sizes:
        LOGGER.debug(
            "Heading scan – size→level mapping: %s; bold-at-body→H%d",
            {f"{s}pt": f"H{l}" for s, l in size_to_level.items()},
            bold_body_level,
        )
    else:
        LOGGER.debug("Heading scan – no large-font headings found, using bold detection only")

    # ------------------------------------------------------------------
    # 5. Build the heading map
    # ------------------------------------------------------------------
    MAX_HEADING_LEN = 80  # ignore long lines – those are paragraphs, not headings
    MIN_HEADING_LEN = 3   # ignore very short lines like "01", "02" section numbers

    heading_map: HeadingMap = {}

    for page_num, font_size, is_bold, text in all_lines:
        normalised = _normalise_text(text)
        if not normalised or len(normalised) > MAX_HEADING_LEN or len(normalised) < MIN_HEADING_LEN:
            continue

        # Skip metric values – standalone percentages, currency amounts,
        # plain numbers, etc.  These appear visually prominent (large
        # font) in infographic / data-heavy pages but are not headings.
        if _is_metric_value(normalised):
            continue

        level: Optional[int] = None

        if font_size in size_to_level:
            level = size_to_level[font_size]
        elif is_bold and font_size >= body_size - 0.5:
            # Bold text at (roughly) body size
            level = bold_body_level

        if level is not None:
            heading_map.setdefault(page_num, []).append((normalised, level))

    LOGGER.info(
        "Heading scan complete: %d pages, %d headings detected",
        len(heading_map),
        sum(len(v) for v in heading_map.values()),
    )
    return heading_map


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_COLLAPSE_WS = re.compile(r"\s+")


# Metric-value pattern: matches standalone percentages, currency
# amounts, plain numbers (optionally with commas/decimals), and
# numbers followed by a short unit (bn, mn, cr, k, m, etc.).
# Examples: "84%", "$10.5 bn", "₹131.4 bn", "233,346", "1.5x"
_METRIC_RE = re.compile(
    r"^"
    r"[₹$€£¥]?\s*"                        # optional leading currency symbol
    r"\d[\d,]*\.?\d*"                       # number (with commas/decimals)
    r"\s*"
    r"(%|bn|mn|cr|k|m|x|bps|pp|GW|MW|hrs|days|lakh|crore)?"  # optional unit
    r"\+?"                                  # optional trailing +
    r"$",
    re.IGNORECASE,
)


def _is_metric_value(text: str) -> bool:
    """
    Return ``True`` if *text* is a standalone metric value rather than
    a real heading.

    Matches patterns like ``84%``, ``$10.5 bn``, ``₹131.4 bn``,
    ``233,346``, ``1.5x``, ``97%+``.
    """
    return bool(_METRIC_RE.match(text.strip()))


def _normalise_text(text: str) -> str:
    """
    Strip and collapse whitespace so heading text can be matched
    reliably between the PDF scan and the markdown output.
    """
    return _COLLAPSE_WS.sub(" ", text.strip())
