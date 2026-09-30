"""Markdown image embedding service.

Handles inserting image references (``![…](…)``) into markdown content
at spatially-accurate positions based on page markers and bounding boxes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class MarkdownImageEmbedder:
    """
    Embeds image URLs into markdown content.

    Handles both PDF (page markers) and HTML (image markers) formats.
    """

    # Regex patterns
    EMBEDDED_PATTERN = re.compile(r'!\[[^\]]*\]\(([a-z]{1,2}[0-9]?://[^)]+/([^/)]+\.(png|jpg|jpeg|webp)))\)', re.IGNORECASE)  # Matches provider URI schemes.
    PAGE_PATTERN = re.compile(r'<!--\s*page_number:\s*(\d+)\s*-->')
    IMAGE_MARKER_PATTERN = re.compile(r'<!--\s*image\s*-->')

    @classmethod
    def embed_image_urls(
        cls,
        markdown: str,
        image_manifest: Dict[str, Any],
    ) -> str:
        """
        Embed cloud storage image URLs into markdown content.

        This method ensures ALL manifest images are embedded in the markdown.
        The markdown conversion may have already embedded some images, so we:
        1. Find which manifest images are already embedded (by filename)
        2. For each page, add any MISSING images at the end of that page

        Args:
            markdown: Original markdown (may already have some images embedded)
            image_manifest: Manifest dict containing images with storage_path

        Returns:
            Markdown with ALL manifest images embedded
        """
        images = image_manifest.get("images", [])
        if not images:
            return markdown

        # Find already-embedded image filenames
        already_embedded = cls._find_embedded_images(markdown)

        # Group manifest images by page
        missing_by_page = cls._group_missing_images(images, already_embedded)

        if not missing_by_page:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CONVERT,
                f"All {len(images)} manifest images already embedded in markdown",
            )
            return markdown

        # Embed images based on marker type
        has_image_markers = bool(cls.IMAGE_MARKER_PATTERN.search(markdown))

        if has_image_markers:
            result, stats = cls._embed_at_image_markers(markdown, missing_by_page)
        else:
            result, stats = cls._embed_at_page_markers(markdown, missing_by_page)

        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT,
            f"Added {stats['total_added']} missing images to markdown. "
            f"Already embedded: {len(already_embedded)}, Total manifest: {len(images)}, "
            f"Pages modified: {sorted(stats['pages_modified'])}",
        )

        return result

    @classmethod
    def _find_embedded_images(cls, markdown: str) -> set:
        """Find filenames of images already embedded in markdown."""
        already_embedded = set()
        for match in cls.EMBEDDED_PATTERN.finditer(markdown):
            already_embedded.add(match.group(1))
        return already_embedded

    @classmethod
    def _group_missing_images(cls, images: List, already_embedded: set) -> Dict[int, List]:
        """Group manifest images by page, excluding already embedded ones."""
        missing_by_page: Dict[int, List] = {}
        for img in images:
            if img is None:
                continue
            storage_path = img.get("storage_path", "")
            if storage_path:
                filename = storage_path.split("/")[-1]
                if filename not in already_embedded:
                    page_num = img.get("page_number", 0)
                    if page_num not in missing_by_page:
                        missing_by_page[page_num] = []
                    missing_by_page[page_num].append(img)

        # Sort images within each page by position
        for page_num in missing_by_page:
            missing_by_page[page_num].sort(key=lambda x: (x.get("bounding_box") or {}).get("y1", 0))

        return missing_by_page

    @classmethod
    def _format_image_markdown(cls, img: Dict) -> Optional[str]:
        """Format a single image as markdown."""
        storage_path = img.get("storage_path", "")
        if not storage_path:
            return None
        description = img.get("description") or ""
        image_type = img.get("image_type", "image")
        image_id = img.get("image_id", "image")
        alt_text = description[:100] if description else f"{image_type.title()} - {image_id}"
        return f"![{alt_text}]({storage_path})"

    @classmethod
    def _embed_at_image_markers(
        cls,
        markdown: str,
        missing_by_page: Dict[int, List],
    ) -> tuple:
        """Embed images at <!-- image --> markers (HTML files).

        Uses regex substitution to handle multiple markers on the same
        line (e.g. when docling outputs the entire document as a table).
        """
        # Create a queue of images sorted by page
        image_queue: List = []
        for page_num in sorted(missing_by_page.keys()):
            for img in missing_by_page[page_num]:
                image_queue.append(img)

        state = {"index": 0, "total_added": 0, "pages_modified": set()}

        def _replace_marker(match: re.Match) -> str:
            """Replace each marker with the marker followed by an image ref."""
            marker_text = match.group(0)
            if state["index"] >= len(image_queue):
                return marker_text
            img = image_queue[state["index"]]
            state["index"] += 1
            img_md = cls._format_image_markdown(img)
            if img_md:
                state["total_added"] += 1
                state["pages_modified"].add(img.get("page_number", 1))
                return f"{marker_text}\n\n{img_md}\n"
            return marker_text

        result = cls.IMAGE_MARKER_PATTERN.sub(_replace_marker, markdown)

        return result, {"total_added": state["total_added"], "pages_modified": state["pages_modified"]}

    @classmethod
    def _embed_at_page_markers(
        cls,
        markdown: str,
        missing_by_page: Dict[int, List],
    ) -> tuple:
        """
        Embed images at spatially-accurate positions within each page.

        For images with bounding-box data the method estimates where in
        the page text each image sits (using the Y-coordinate relative to
        page height) and inserts the ``![…](…)`` reference **between the
        correct paragraphs** rather than appending at the end of the page.

        Falls back to end-of-page insertion when no bounding boxes are
        available.
        """
        # Split markdown into per-page blocks
        page_blocks = cls._split_markdown_by_pages(markdown)

        if not page_blocks:
            # No page markers found – append images at end
            return cls._append_images_at_end(markdown, missing_by_page)

        total_added = 0
        pages_modified: set = set()
        result_parts: List[str] = []

        for page_num, page_content in page_blocks:
            page_images = missing_by_page.pop(page_num, [])

            if page_images:
                page_content, count = cls._insert_images_spatially(page_content, page_images)
                total_added += count
                if count:
                    pages_modified.add(page_num)

            result_parts.append(f"<!-- page_number: {page_num} -->")
            result_parts.append(page_content)

        # Any remaining pages that had no marker
        for page_num in sorted(missing_by_page.keys()):
            for img in missing_by_page[page_num]:
                img_md = cls._format_image_markdown(img)
                if img_md:
                    result_parts.append("")
                    result_parts.append(img_md)
                    total_added += 1
                    pages_modified.add(page_num)

        return "\n\n".join(result_parts), {
            "total_added": total_added,
            "pages_modified": pages_modified,
        }

    # ------------------------------------------------------------------
    # Spatial insertion helpers
    # ------------------------------------------------------------------

    @classmethod
    def _split_markdown_by_pages(cls, markdown: str) -> List:
        """Split *markdown* into ``[(page_num, content), …]`` using page markers."""
        parts = cls.PAGE_PATTERN.split(markdown)
        pages: List = []
        # parts alternates: [pre, page_num_str, content, page_num_str, content, …]
        i = 1
        while i < len(parts) - 1:
            page_num = int(parts[i])
            content = parts[i + 1].strip()
            pages.append((page_num, content))
            i += 2
        return pages

    @classmethod
    def _insert_images_spatially(
        cls,
        page_content: str,
        images: List[Dict],
    ) -> tuple:
        """
        Insert *images* into *page_content* at positions matching their
        bounding-box Y-coordinates.

        Returns ``(new_content, images_inserted_count)``.
        """
        if not images:
            return page_content, 0

        # Separate images with/without spatial data
        spatial_images: List = []
        tail_images: List = []
        for img in images:
            bbox = img.get("bounding_box") or {}
            if bbox.get("y1") is not None:
                spatial_images.append(img)
            else:
                tail_images.append(img)

        if not spatial_images and not tail_images:
            return page_content, 0

        # Split content into blocks (blank-line delimited)
        blocks = re.split(r"\n{2,}", page_content)
        blocks = [b for b in blocks if b.strip()]  # drop empty blocks

        if not blocks:
            # Page is empty – just emit images
            parts: List[str] = []
            count = 0
            for img in spatial_images + tail_images:
                md = cls._format_image_markdown(img)
                if md:
                    parts.append(md)
                    count += 1
            return "\n\n".join(parts), count

        # Determine page height
        page_height = cls._get_page_height_from_images(spatial_images)

        # Build insertion plan
        insert_after: Dict[int, List[str]] = {}
        num_blocks = len(blocks)

        for img in spatial_images:
            y1 = (img.get("bounding_box") or {}).get("y1", 0)
            frac = y1 / page_height if page_height > 0 else 0
            block_idx = min(int(frac * num_blocks), num_blocks - 1)
            insert_after.setdefault(block_idx, [])
            md = cls._format_image_markdown(img)
            if md:
                insert_after[block_idx].append(md)

        # Reassemble page with images interleaved
        new_parts: List[str] = []
        count = 0
        for idx, block in enumerate(blocks):
            new_parts.append(block)
            if idx in insert_after:
                for img_md in insert_after[idx]:
                    new_parts.append(img_md)
                    count += 1

        # Append tail (no-bbox) images at end
        for img in tail_images:
            md = cls._format_image_markdown(img)
            if md:
                new_parts.append(md)
                count += 1

        return "\n\n".join(new_parts), count

    @classmethod
    def _get_page_height_from_images(cls, images: List[Dict]) -> float:
        """Derive page height from image metadata (fallback: 792 US-Letter points)."""
        for img in images:
            ph = img.get("page_height")
            if ph and ph > 0:
                return float(ph)
        return 792.0

    @classmethod
    def _append_images_at_end(
        cls,
        markdown: str,
        missing_by_page: Dict[int, List],
    ) -> tuple:
        """Fallback: append all images at the end of the document."""
        lines = [markdown]
        total_added = 0
        pages_modified: set = set()

        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT,
            f"No page markers found. "
            f"Appending {sum(len(v) for v in missing_by_page.values())} images at end.",
        )

        for page_num in sorted(missing_by_page.keys()):
            for img in missing_by_page[page_num]:
                img_md = cls._format_image_markdown(img)
                if img_md:
                    lines.append("")
                    lines.append(img_md)
                    total_added += 1
                    pages_modified.add(page_num)

        return "\n".join(lines), {
            "total_added": total_added,
            "pages_modified": pages_modified,
        }

    # ------------------------------------------------------------------
    # Context text enrichment
    # ------------------------------------------------------------------

    # Number of non-empty text lines to capture before/after an image position
    _CONTEXT_LINES = 3

    @classmethod
    def enrich_manifest_with_context(
        cls,
        markdown: str,
        image_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich each image in the manifest with surrounding text context.

        For documents with ``<!-- image -->`` markers the text lines
        immediately before and after each marker are captured.  For
        PDF-style documents (``<!-- page_number: N -->``) the text
        surrounding each embedded ``![…](…)`` reference is used instead.

        The result is stored in each image dict as ``context_text``.

        Args:
            markdown: The final markdown content (after image embedding).
            image_manifest: Image manifest dict (modified in-place).

        Returns:
            The same *image_manifest* dict (for chaining convenience).
        """
        images = image_manifest.get("images", [])
        if not images:
            return image_manifest

        has_image_markers = bool(cls.IMAGE_MARKER_PATTERN.search(markdown))

        if has_image_markers:
            cls._enrich_context_from_markers(markdown, images)
        else:
            cls._enrich_context_from_embedded(markdown, images)

        enriched = sum(1 for img in images if img.get("context_text"))
        if enriched:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CONVERT,
                f"Enriched {enriched}/{len(images)} images with surrounding context text",
            )

        return image_manifest

    @classmethod
    def _enrich_context_from_markers(
        cls,
        markdown: str,
        images: List[Dict[str, Any]],
    ) -> None:
        """Extract context around ``<!-- image -->`` markers (HTML-origin docs).

        Uses ``re.finditer`` to locate each marker occurrence so that
        multiple markers on the same line (e.g. table-wrapped documents)
        are handled correctly.
        """
        marker_positions: List[int] = [
            m.start() for m in cls.IMAGE_MARKER_PATTERN.finditer(markdown)
        ]

        for img_idx, img in enumerate(images):
            if img_idx >= len(marker_positions):
                break
            pos = marker_positions[img_idx]
            context = cls._extract_surrounding_text_by_offset(markdown, pos)
            if context:
                img["context_text"] = context

    @classmethod
    def _enrich_context_from_embedded(
        cls,
        markdown: str,
        images: List[Dict[str, Any]],
    ) -> None:
        """Extract context around embedded ``![…](…)`` references (PDF docs)."""
        lines = markdown.split("\n")

        for img in images:
            storage_path = img.get("storage_path", "")
            if not storage_path:
                continue
            filename = storage_path.split("/")[-1]
            for i, line in enumerate(lines):
                if filename in line and line.strip().startswith("!["):
                    context = cls._extract_surrounding_text(lines, i)
                    if context:
                        img["context_text"] = context
                    break

    @classmethod
    def _extract_surrounding_text(
        cls,
        lines: List[str],
        anchor_idx: int,
    ) -> str:
        """
        Collect up to ``_CONTEXT_LINES`` non-empty text lines before and
        after *anchor_idx*, skipping image markers, image refs, and blanks.
        """
        before: List[str] = []
        after: List[str] = []

        # Collect lines before
        i = anchor_idx - 1
        while i >= 0 and len(before) < cls._CONTEXT_LINES:
            line = lines[i].strip()
            if line and not cls._is_image_or_marker(line):
                before.append(line)
            i -= 1
        before.reverse()

        # Collect lines after
        i = anchor_idx + 1
        while i < len(lines) and len(after) < cls._CONTEXT_LINES:
            line = lines[i].strip()
            if line and not cls._is_image_or_marker(line):
                after.append(line)
            i += 1

        combined = " ".join(before + after).strip()
        return combined[:500] if combined else ""

    # Pattern to split text into meaningful segments for offset-based context
    _SEGMENT_SPLIT_PATTERN = re.compile(
        r'(?:<!--\s*image\s*-->|!\[[^\]]*\]\([^)]+\))'
    )

    @classmethod
    def _extract_surrounding_text_by_offset(
        cls,
        text: str,
        marker_start: int,
    ) -> str:
        """
        Extract surrounding text context around a marker at a character
        offset.  Works for inline markers (multiple on a single line).

        Looks at text before and after the marker, splits on sentence-like
        boundaries (``**…**``, headings, double spaces used by docling),
        and returns nearby segments.
        """
        marker_match = cls.IMAGE_MARKER_PATTERN.match(text, marker_start)
        marker_end = marker_match.end() if marker_match else marker_start + 16

        # Get text windows before and after the marker
        window = 500
        before_text = text[max(0, marker_start - window):marker_start]
        after_text = text[marker_end:marker_end + window]

        # Remove other image markers/refs from the windows
        before_text = cls._SEGMENT_SPLIT_PATTERN.sub('', before_text)
        after_text = cls._SEGMENT_SPLIT_PATTERN.sub('', after_text)

        # Split on common segment boundaries (double space from docling,
        # markdown headings, or sentence endings)
        segment_pattern = re.compile(r'(?:  |\n|(?<=\.)\s+|(?<=\|)\s*)')

        before_segments = [
            s.strip() for s in segment_pattern.split(before_text) if s.strip()
        ]
        after_segments = [
            s.strip() for s in segment_pattern.split(after_text) if s.strip()
        ]

        # Take last N segments before and first N after
        n = cls._CONTEXT_LINES
        before_parts = before_segments[-n:] if before_segments else []
        after_parts = after_segments[:n] if after_segments else []

        combined = " ".join(before_parts + after_parts).strip()
        # Clean up markdown formatting noise
        combined = re.sub(r'\s+', ' ', combined)
        return combined[:500] if combined else ""

    @classmethod
    def _is_image_or_marker(cls, line: str) -> bool:
        """Return True if *line* is an image reference or HTML marker."""
        if cls.IMAGE_MARKER_PATTERN.search(line):
            return True
        if cls.PAGE_PATTERN.search(line):
            return True
        if line.startswith("![") and "](" in line:
            return True
        return False
