"""Shared page-marker utilities for chunking adapters.

Provides a single compiled regex and helper functions for working with
``<!-- page_number: N -->`` markers embedded in markdown documents.
"""
from __future__ import annotations

import re
from typing import List, Optional

# Matches e.g. ``<!-- page_number: 42 -->``
PAGE_MARKER_RE = re.compile(r"<!--\s*page_number:\s*(\d+)\s*-->", re.IGNORECASE)

# Matches e.g. ``<!-- slide_type: title -->`` (PPTX only, emitted by the
# Vision LLM right after the page_number marker).
SLIDE_TYPE_MARKER_RE = re.compile(
    r"<!--\s*page_number:\s*(\d+)\s*-->\s*<!--\s*slide_type:\s*(title|content)\s*-->",
    re.IGNORECASE,
)


def extract_slide_type_map(text: str) -> dict[str, str]:
    """Return ``{page_number: slide_type}`` for every PPTX slide-type marker
    pair found in *text*. Pages without a slide_type marker are omitted."""
    return {page: stype.lower() for page, stype in SLIDE_TYPE_MARKER_RE.findall(text)}


def extract_page_number(text: str) -> Optional[str]:
    """Return the first page-marker value found in *text*, or ``None``."""
    m = PAGE_MARKER_RE.search(text)
    return m.group(1) if m else None


def extract_all_page_numbers(text: str) -> List[str]:
    """Return all unique page-marker values found in *text*, ordered."""
    return list(dict.fromkeys(PAGE_MARKER_RE.findall(text)))


def strip_page_markers(text: str) -> str:
    """Remove ``<!-- page_number: N -->`` markers and collapse extra blank lines."""
    cleaned = PAGE_MARKER_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
