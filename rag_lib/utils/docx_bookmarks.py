"""DOCX Bookmark Injection Utility.

Injects Word bookmarks at heading paragraphs in a DOCX file so that
the Office Online viewer (or desktop Word) can navigate directly to a
section via ``wdNavigateTo=<bookmark_name>``.

Bookmark names are slugified heading text (max 40 chars, alphanumeric +
underscores).  A counter suffix is added when duplicates arise.

Usage:
    from rag_lib.utils.docx_bookmarks import inject_heading_bookmarks

    bookmark_map = inject_heading_bookmarks(
        src_path="input.docx",
        dst_path="output_with_bookmarks.docx",
    )
    # bookmark_map: {"introduction": "Introduction", "scope_and_objectives": "Scope and Objectives", ...}
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)

# Max bookmark name length in Word
_MAX_BOOKMARK_LEN = 40


def _slugify(text: str) -> str:
    """Convert heading text to a valid Word bookmark name.

    Word bookmark rules:
    - Must start with a letter or underscore
    - Can contain letters, digits, underscores
    - Max 40 characters
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Replace non-alphanumeric with underscore
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())
    # Strip leading/trailing underscores
    text = text.strip("_")
    # Must start with a letter
    if text and not text[0].isalpha():
        text = "s_" + text
    # Truncate
    if len(text) > _MAX_BOOKMARK_LEN:
        text = text[:_MAX_BOOKMARK_LEN].rstrip("_")
    return text or "untitled"


def inject_heading_bookmarks(
    src_path: str,
    dst_path: Optional[str] = None,
) -> Dict[str, str]:
    """Inject bookmarks at every heading paragraph in a DOCX file.

    Args:
        src_path: Path to the source .docx file.
        dst_path: Path to save the bookmarked .docx.  If None, overwrites src_path.

    Returns:
        Mapping of bookmark_name → original heading text.
        Empty dict if python-docx is not installed or the file has no headings.
    """
    try:
        from docx import Document
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except ImportError:
        LOGGER.warning("python-docx not installed; bookmark injection skipped")
        return {}

    if dst_path is None:
        dst_path = src_path

    doc = Document(src_path)
    bookmark_map: Dict[str, str] = {}
    slug_counts: Dict[str, int] = {}
    bookmark_id = 0

    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower()
        # Match heading styles: "Heading 1", "Heading 2", etc.
        if not style_name.startswith("heading"):
            continue

        heading_text = para.text.strip()
        if not heading_text:
            continue

        # Generate unique bookmark slug
        slug = _slugify(heading_text)
        if slug in slug_counts:
            slug_counts[slug] += 1
            slug = f"{slug}_{slug_counts[slug]}"
        else:
            slug_counts[slug] = 0

        # Inject bookmark XML before the paragraph's first run
        # <w:bookmarkStart w:id="N" w:name="slug"/>
        # <w:bookmarkEnd w:id="N"/>
        bookmark_id += 1

        bm_start = OxmlElement("w:bookmarkStart")
        bm_start.set(qn("w:id"), str(bookmark_id))
        bm_start.set(qn("w:name"), slug)

        bm_end = OxmlElement("w:bookmarkEnd")
        bm_end.set(qn("w:id"), str(bookmark_id))

        # Insert at beginning of paragraph element
        para._element.insert(0, bm_start)
        # Insert end right after start
        bm_start.addnext(bm_end)

        bookmark_map[slug] = heading_text

    if bookmark_map:
        doc.save(dst_path)
        LOGGER.info(
            "Injected %d bookmarks into %s → %s",
            len(bookmark_map), src_path, dst_path,
        )
    else:
        LOGGER.info("No headings found in %s; no bookmarks injected", src_path)
        # Still save to dst if different from src
        if dst_path != src_path:
            import shutil
            shutil.copy2(src_path, dst_path)

    return bookmark_map


def heading_to_bookmark_slug(heading_text: str, existing_slugs: Optional[Dict[str, int]] = None) -> str:
    """Convert a heading text to the same bookmark slug that would be injected.

    This is used at chunking time to generate the ``anchor_id`` metadata
    field that matches the bookmark name in the DOCX.

    Args:
        heading_text: The heading text (from markdown ## heading).
        existing_slugs: Optional dict tracking slug counts for deduplication.
            If provided, it will be mutated to track the count.

    Returns:
        Bookmark slug string.
    """
    slug = _slugify(heading_text)
    if existing_slugs is not None:
        if slug in existing_slugs:
            existing_slugs[slug] += 1
            slug = f"{slug}_{existing_slugs[slug]}"
        else:
            existing_slugs[slug] = 0
    return slug
