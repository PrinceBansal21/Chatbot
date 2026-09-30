"""
Hybrid Markdown Adapter

This adapter combines the strengths of multiple conversion approaches:
1. Docling - Excellent for tables, structured text, and document layout
2. Vision LLM - Excellent for infographics, charts, and visual data displays

Strategy:
- Analyze each page to determine if it's primarily text/tables or infographic
- **Correct page orientation** - automatically fix rotated pages
- Use Docling for text-heavy/table pages
- Use Vision LLM for infographic/visual pages
- Merge results into coherent markdown

This is ideal for business documents like:
- Financial datasheets with both tables and KPI infographics
- Annual reports with charts and structured data
- Marketing materials with visual layouts
- Presentations converted to PDF
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_lib.interfaces.markdown_manager import MarkdownManager

LOGGER = logging.getLogger(__name__)

# Try to import dependencies
try:
    import pypdfium2 as pdfium

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False

try:
    import pdfplumber as _pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Import orientation correction utilities
try:
    from rag_lib.adapters.markdown.utils.pdf_orientation import (
        analyze_page_for_rotation,
    )

    ORIENTATION_UTILS_AVAILABLE = True
except ImportError:
    ORIENTATION_UTILS_AVAILABLE = False
    LOGGER.debug("PDF orientation utilities not available")


class HybridAdapter(MarkdownManager):
    """
    Hybrid markdown adapter that intelligently combines Docling and Vision LLM.

    Page classification:
    - Pages with detected tables → Use Docling
    - Pages with high image/graphic ratio → Use Vision LLM
    - Pages with minimal extractable text → Use Vision LLM
    - Default → Use Docling
    """

    def __init__(self, **config):
        """
        Initialize the Hybrid adapter.

        Args:
            **config: Configuration options:
                - vision_model: str - Model for Vision LLM (nova-pro, claude-sonnet)
                - region: str - Cloud region for the configured model provider
                - infographic_pages: List[int] - Force Vision LLM for specific pages
                - table_pages: List[int] - Force Docling for specific pages
                - text_threshold: float - Min text coverage to use Docling (0-1, default 0.3)
                - always_vision_first_page: bool - Use Vision LLM for page 1 (default True)
                - correct_orientation: bool - Auto-correct rotated pages (default True)
                - docling_config: dict - Config to pass to Docling adapter
                - vision_config: dict - Config to pass to Vision LLM adapter
                - merge_strategy: str - How to merge results (sequential, smart)
        """
        self.config = config

        # Page routing configuration
        self.infographic_pages = set(config.get("infographic_pages", []))
        self.table_pages = set(config.get("table_pages", []))
        self.text_threshold = config.get("text_threshold", 0.3)
        self.always_vision_first_page = config.get("always_vision_first_page", True)

        # Page orientation correction
        self.correct_orientation = config.get("correct_orientation", True)

        # Initialize adapters lazily
        self._docling_adapter = None
        self._vision_adapter = None

        # Sub-adapter configurations
        self.docling_config = config.get("docling_config", {})
        self.vision_config = config.get("vision_config", {})

        # Pass correct_orientation to vision adapter
        if self.correct_orientation:
            self.vision_config.setdefault("correct_orientation", True)

        # Apply common config to vision adapter
        if "region" in config:
            self.vision_config.setdefault("region", config["region"])
        if "vision_model" in config:
            self.vision_config.setdefault("model_id", config["vision_model"])

        LOGGER.info(
            "HybridAdapter initialized: vision_model=%s, always_vision_first_page=%s, correct_orientation=%s",
            self.vision_config.get("model", "nova-pro"),
            self.always_vision_first_page,
            self.correct_orientation,
        )

    @property
    def docling_adapter(self):
        """Lazy initialization of Docling adapter."""
        if self._docling_adapter is None:
            from rag_lib.adapters.markdown.docling import DoclingAdapter

            self._docling_adapter = DoclingAdapter(**self.docling_config)
        return self._docling_adapter

    @property
    def vision_adapter(self):
        """Lazy initialization of Vision LLM adapter."""
        if self._vision_adapter is None:
            from rag_lib.adapters.markdown.vision_llm import VisionLLMAdapter

            self._vision_adapter = VisionLLMAdapter(**self.vision_config)
        return self._vision_adapter

    def _analyze_page(self, plumber_pdf, page_num: int) -> str:
        """
        Analyze a PDF page and determine the best extraction method.

        Args:
            plumber_pdf: pdfplumber PDF object
            page_num: Page number (0-indexed)

        Returns:
            'vision' or 'docling'
        """
        # Decision logic for forced pages (no PDF parsing needed)
        if (page_num + 1) in self.infographic_pages:
            return "vision"
        if (page_num + 1) in self.table_pages:
            return "docling"
        if page_num == 0 and self.always_vision_first_page:
            return "vision"

        try:
            page = plumber_pdf.pages[page_num]
            page_area = page.width * page.height
            if page_area <= 0:
                return "docling"

            # Text area from words
            words = page.extract_words() or []
            text_area = sum(
                (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
            )

            # Image area from page images
            images = page.images or []
            image_area = sum(
                (img["x1"] - img["x0"]) * (img["bottom"] - img["top"]) for img in images
            )

            # Table detection
            tables = page.find_tables() or []
            has_tables = len(tables) > 0

            text_coverage = text_area / page_area if page_area > 0 else 0
            image_coverage = image_area / page_area if page_area > 0 else 0

            LOGGER.debug(
                "Page %d analysis: text_coverage=%.2f, image_coverage=%.2f, tables=%s",
                page_num + 1,
                text_coverage,
                image_coverage,
                has_tables,
            )

            if has_tables:
                return "docling"
            if image_coverage > 0.3 and text_coverage < self.text_threshold:
                return "vision"
            if text_coverage < 0.1:
                return "vision"
            return "docling"
        except Exception:
            return "docling"

    def _extract_page_with_docling(
        self, file_path: str, page_nums: List[int]
    ) -> Dict[int, str]:
        """
        Extract specific pages using Docling.

        Note: Docling processes the entire document. When page-level extraction
        isn't supported, we'll use PyMuPDF to help identify page boundaries.
        """
        result = self.docling_adapter._converter.convert(Path(file_path))

        page_contents = {}

        # Get the full markdown
        full_md = result.document.export_to_markdown()

        import re

        # First, try to extract page-level content from the document object
        if hasattr(result.document, "pages") and result.document.pages:
            pages_map = result.document.pages
            LOGGER.debug(
                "Docling pages_map type: %s, keys: %s",
                type(pages_map),
                list(pages_map.keys()) if isinstance(pages_map, dict) else "N/A",
            )

            if isinstance(pages_map, dict):
                for page_no in page_nums:
                    if page_no in pages_map:
                        page_obj = pages_map[page_no]
                        if hasattr(page_obj, "export_to_markdown"):
                            page_contents[page_no] = page_obj.export_to_markdown()
                        else:
                            LOGGER.debug(
                                "Page %d object has no export_to_markdown method",
                                page_no,
                            )

        # If page-level extraction succeeded for all requested pages, return
        if len(page_contents) == len(page_nums):
            return page_contents

        # Fallback: Since we can't reliably split Docling output by page,
        # use Vision LLM for these pages instead (more reliable page-level extraction)
        LOGGER.warning(
            "Docling page-level extraction not available. "
            "Falling back to Vision LLM for pages: %s",
            page_nums,
        )
        return self._extract_page_with_vision(file_path, page_nums)

    def _is_content_blocked(self, content: str) -> bool:
        """Check if content was blocked by Vision LLM content filters."""
        if not content:
            return True

        blocked_indicators = [
            "blocked by our content filters",
            "content has been blocked",
            "cannot process this content",
            "content filters",
        ]

        content_lower = content.lower()
        return any(
            indicator.lower() in content_lower for indicator in blocked_indicators
        )

    def _extract_blocked_pages_with_docling(
        self, file_path: str, page_nums: List[int]
    ) -> Dict[int, str]:
        """
        Extract specific blocked pages using Docling with page_range parameter.

        Uses Docling's page_range to convert only the specific pages,
        preserving table structure and formatting.
        """
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
            TableFormerMode,
            EasyOcrOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        page_contents = {}

        # Create a converter with table extraction settings
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True, mode=TableFormerMode.ACCURATE
            ),
            do_ocr=True,
            ocr_options=EasyOcrOptions(lang=["en"]),
        )

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        converter = DocumentConverter(format_options=format_options)

        for page_num in page_nums:
            LOGGER.info(
                "Extracting blocked page %d with Docling (page_range)", page_num
            )

            try:
                # Use Docling's page_range parameter to extract only this page
                result = converter.convert(
                    Path(file_path), page_range=(page_num, page_num)
                )

                # Export the single page to markdown
                page_md = result.document.export_to_markdown()

                if page_md and page_md.strip():
                    page_contents[page_num] = page_md.strip()
                    LOGGER.debug(
                        "Successfully extracted page %d with Docling", page_num
                    )
                else:
                    LOGGER.warning(
                        "Docling returned empty content for page %d", page_num
                    )
                    page_contents[page_num] = (
                        f"<!-- Page {page_num} - No content extracted -->"
                    )

            except Exception as e:
                LOGGER.error("Failed to extract page %d with Docling: %s", page_num, e)
                page_contents[page_num] = (
                    f"<!-- Page {page_num} - Extraction error: {e} -->"
                )

        return page_contents

    def _extract_page_with_vision(
        self, file_path: str, page_nums: List[int]
    ) -> Dict[int, str]:
        """Extract specific pages using Vision LLM, with Docling fallback for blocked content."""
        page_contents = {}
        blocked_pages = []

        # Render pages to images using pypdfium2
        pdf = pdfium.PdfDocument(file_path)
        dpi = self.vision_config.get("dpi", 150)
        scale = dpi / 72

        try:
            for page_num in page_nums:
                if page_num <= 0 or page_num > len(pdf):
                    continue

                page = pdf[page_num - 1]  # 0-indexed

                # pypdfium2 handles /Rotate automatically during rendering
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                # Extract content using vision adapter
                content = self.vision_adapter._extract_page_content(img_bytes, page_num)

                # Check if content was blocked by content filters
                if self._is_content_blocked(content):
                    LOGGER.warning(
                        "Page %d content blocked by Vision LLM filters, "
                        "will use Docling fallback",
                        page_num,
                    )
                    blocked_pages.append(page_num)
                else:
                    page_contents[page_num] = content
        finally:
            pdf.close()

        # For blocked pages, extract using Docling with proper table extraction
        if blocked_pages:
            LOGGER.info(
                "Extracting %d blocked pages with Docling: %s",
                len(blocked_pages),
                blocked_pages,
            )
            docling_content = self._extract_blocked_pages_with_docling(
                file_path, blocked_pages
            )
            page_contents.update(docling_content)

        return page_contents

    def _extract_remaining_pages_pymupdf(
        self, file_path: str, page_nums: List[int], page_contents: Dict[int, str]
    ) -> None:
        """
        Extract remaining pages using pdfplumber with table detection.
        Updates page_contents in place.
        """
        pdf = _pdfplumber.open(file_path)

        try:
            for page_num in page_nums:
                if page_num <= 0 or page_num > len(pdf.pages):
                    continue

                page = pdf.pages[page_num - 1]
                markdown_parts = []

                # Try to extract tables first
                try:
                    tables = page.find_tables()
                    if tables:
                        for table in tables:
                            table_md = self._table_to_markdown(table)
                            if table_md:
                                markdown_parts.append(table_md)
                except Exception as e:
                    LOGGER.debug("Table extraction failed for page %d: %s", page_num, e)

                # Get remaining text
                text = page.extract_text() or ""
                if text.strip():
                    if markdown_parts:
                        markdown_parts.insert(0, text.strip())
                    else:
                        markdown_parts.append(text.strip())

                if markdown_parts:
                    page_contents[page_num] = "\n\n".join(markdown_parts)
                else:
                    page_contents[page_num] = (
                        f"<!-- Page {page_num} - No extractable content -->"
                    )
        finally:
            pdf.close()

    def _table_to_markdown(self, table) -> str:
        """Convert a pdfplumber table to markdown format."""
        try:
            data = table.extract()
            if not data or not data[0]:
                return ""

            # Build markdown table
            lines = []

            # Header row
            header = data[0]
            header_cells = [str(cell) if cell else "" for cell in header]
            lines.append("| " + " | ".join(header_cells) + " |")

            # Separator
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")

            # Data rows
            for row in data[1:]:
                cells = [str(cell) if cell else "" for cell in row]
                lines.append("| " + " | ".join(cells) + " |")

            return "\n".join(lines)
        except Exception as e:
            LOGGER.debug("Failed to convert table to markdown: %s", e)
            return ""

    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """
        Convert a document to markdown using the hybrid approach.

        Args:
            file_path: Path to the document
            mime_type: Optional MIME type
            **kwargs: Additional options:
                - force_method: str - Force 'vision' or 'docling' for all pages

        Returns:
            Markdown content
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # For non-PDF files, delegate to appropriate adapter
        if suffix != ".pdf":
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                return self.vision_adapter.convert(file_path, mime_type, **kwargs)
            else:
                return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        # Check for forced method
        force_method = kwargs.get("force_method")
        if force_method == "vision":
            return self.vision_adapter.convert(file_path, mime_type, **kwargs)
        elif force_method == "docling":
            return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        if not PYPDFIUM2_AVAILABLE:
            LOGGER.warning("pypdfium2 not available, falling back to Docling only")
            return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        # Analyze pages and route to appropriate adapter using pdfplumber
        plumber = _pdfplumber.open(file_path)

        try:
            num_pages = len(plumber.pages)
            page_methods = {}

            for page_num in range(num_pages):
                method = self._analyze_page(plumber, page_num)
                page_methods[page_num + 1] = method  # 1-indexed

            LOGGER.info("Page routing: %s", {p: m for p, m in page_methods.items()})
        finally:
            plumber.close()

        # Group pages by method
        vision_pages = [p for p, m in page_methods.items() if m == "vision"]
        docling_pages = [p for p, m in page_methods.items() if m == "docling"]

        # Extract content
        all_content = {}

        if docling_pages:
            LOGGER.info(
                "Processing %d pages with Docling: %s",
                len(docling_pages),
                docling_pages,
            )
            docling_content = self._extract_page_with_docling(file_path, docling_pages)
            all_content.update(docling_content)

        if vision_pages:
            LOGGER.info(
                "Processing %d pages with Vision LLM: %s",
                len(vision_pages),
                vision_pages,
            )
            vision_content = self._extract_page_with_vision(file_path, vision_pages)
            all_content.update(vision_content)

        # Merge content in page order
        markdown_parts = []
        for page_num in sorted(all_content.keys()):
            markdown_parts.append(f"<!-- page_number: {page_num} -->")
            markdown_parts.append(all_content[page_num].strip())

        return "\n\n".join(markdown_parts)
