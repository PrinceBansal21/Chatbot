"""
Vision LLM Markdown Adapter

This adapter converts documents (especially PDFs with infographics, charts, and
complex layouts) to markdown using a Vision LLM model. It renders each page as
an image and sends it to a multimodal model for extraction.

Supported models:
- Provider-configured multimodal models through cloud_lib
- GPT-4o via OpenAI (requires API key)

This adapter is best for:
- Infographic-heavy documents (datasheets, brochures, presentations)
- Documents with complex visual layouts
- Charts, diagrams, and visual data representations
- Documents where text extraction loses spatial relationships
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rag_lib.interfaces.markdown_manager import MarkdownManager

LOGGER = logging.getLogger(__name__)

# Default prompt for markdown extraction
DEFAULT_EXTRACTION_PROMPT = """\
You are a document page analysis expert. Convert this page image into \
well-structured Markdown that captures **every piece of visible content**.

RULES:
1. **Headings**: Use `## Section Title` for main sections on this page. \
   Use `### Sub-section` and `####` as needed for nested groups. \
   NEVER use `#` (H1) — it is reserved for the document title only. \
   NEVER use headings for individual metrics, values, or data points.
2. **Sections / grouped regions**: Use `## Section Name` for each distinct \
   content group, box, or labelled region on the page.
3. **Bullet points**: Reproduce every bullet exactly. Use nested `-` for \
   sub-bullets.  Pair labels with values on the same line: \
   `- **Label:** value`.
4. **Tables**: Convert tabular content to proper Markdown tables.
5. **Diagrams / flowcharts / process maps**: Describe the structure. \
   List each node/segment as a heading or bullet, preserving parent-child \
   or sequential relationships.
6. **Charts / graphs**: Describe the data with actual values as bullet \
   points under a descriptive heading.
7. **Images / logos**: Mention them briefly (e.g., "Logo: Company Name") \
   but do NOT invent data.
8. **Annotations & footnotes**: Capture notes, disclaimers, fine print, \
   and footer text.
9. Do NOT add commentary or preamble — output raw Markdown only.

SPATIAL POSITIONING (required):
- For EVERY `##` section heading, append its position on the page in \
  parentheses using one of these labels: \
  `(top-left)`, `(top-center)`, `(top-right)`, \
  `(center-left)`, `(center)`, `(center-right)`, \
  `(bottom-left)`, `(bottom-center)`, `(bottom-right)`, \
  `(left-column)`, `(right-column)`, `(full-width)`, `(spanning)`.
- For diagrams or images, note their position: e.g., `(center of page)`.
- Example: `## Safety Monitoring & Engagement (top-left)`

REGION-BOUNDARY ACCURACY (critical):
- Each labelled region on the page may have clear visual boundaries \
  (coloured background, border, or header bar).
- Assign every bullet point ONLY to the region it is visually inside. \
  Look carefully at the region edges — adjacent regions may share a border.
- If a bullet is near the boundary between two regions, check which \
  header/title bar it falls under, not which region is nearest.
- NEVER move a bullet from one region to another.

Formatting example:
```
## Financial Highlights (top-left)

### Revenue Performance
- **Revenue:** $10.5 bn ((-) 2.3% YoY)
- **Operating Margin:** 17.1%

## Workforce Statistics (top-right)
- **Total Employees:** 233,346

## Process Flow (center)
- Step A → Step B → Step C

> Note: Figures are unaudited. (bottom-center)
```

Extract everything visible on this page using the format above:\
"""

# Prompt for pages that appear to be infographic-heavy
INFOGRAPHIC_PROMPT = """\
You are analysing an **infographic or diagram page** from a document. \
Extract ALL visible information into structured Markdown.

HEADING HIERARCHY:
- `##`  — Each distinct box, region, or labelled group on the page
- `###` / `####` — Nested sub-groups within a region
- NEVER use `#` (H1) — it is reserved for the document title only
- NEVER use headings for individual metrics, numbers, or values — \
  individual metrics MUST be bullet points

SPATIAL POSITIONING (required):
- For EVERY `##` section heading, append its position on the page in \
  parentheses using one of these labels: \
  `(top-left)`, `(top-center)`, `(top-right)`, \
  `(center-left)`, `(center)`, `(center-right)`, \
  `(bottom-left)`, `(bottom-center)`, `(bottom-right)`, \
  `(left-column)`, `(right-column)`, `(full-width)`, `(spanning)`.
- For diagrams or central visuals, note their position: \
  e.g., `## Diagram (center)`.
- Example: `## IT Infrastructure & Administration (top-right)`

CRITICAL RULES:
1. Every number/value MUST appear on the SAME bullet as its label: \
   `- **Label:** value`
2. Include percentages, comparisons, and units in context: \
   `- **IT Services Revenue:** $10.5 bn ((-) 2.3% YoY CC)`
3. For circular / flow diagrams: list each segment as a heading, then \
   its detail bullets.
4. For connected components: describe the connections (e.g., \
   "→", "feeds into", "part of").
5. Group related metrics together under ONE section heading.
6. Use tables ONLY for truly tabular/comparative data with multiple columns.
7. Capture ALL footnotes, disclaimers, and fine print.
8. Do NOT invent data — only report what is visible.

REGION-BOUNDARY ACCURACY (critical):
- Each labelled region has clear visual boundaries \
  (coloured background, border, or header bar).
- Assign every bullet point ONLY to the region it is visually inside. \
  Look carefully at the region edges — adjacent regions may share a border.
- If a bullet is near the boundary between two regions, check which \
  header/title bar it falls under, not which region is nearest.
- NEVER move a bullet from one region to another.

Example output format:
```
## Year at a Glance (top-center)

### Financial Performance
- **IT Services Revenue:** $10.5 bn ((-) 2.3% YoY CC)
- **Total Bookings TCV:** $14.3 bn ((-) 3.8% YoY CC)
- **Large Deal TCV:** $5.4 bn (+17.5% YoY CC)
- **Operating Margin:** 17.1% (+0.9% YoY)

## Workforce (center-right)
- **Total Employees:** 233,346
- **Women Employees:** 37.1%

## Process Flow (center)
- Node A → Node B → Node C
```

Output raw Markdown only — no preamble or commentary.\
"""


class VisionLLMAdapter(MarkdownManager):
    """
    Markdown adapter using Vision LLM for document conversion.

    This adapter renders document pages as images and uses a multimodal
    LLM to extract structured markdown content, preserving spatial
    relationships between labels and values.
    """

    SUPPORTED_MODELS = None  # Populated lazily in __init__

    @classmethod
    def _get_supported_models(cls) -> dict:
        """Load supported models from central config. Fails if not configured."""
        from rag_lib.model_config import get_vision_llm_models

        models = get_vision_llm_models()
        if not models:
            from rag_lib.model_config import ModelConfigError

            raise ModelConfigError(
                "VisionLLMAdapter: vision_llm_models not configured. "
                "Ensure the ai-models cloud config is available to the pipeline."
            )
        return models

    def __init__(self, **config):
        """
        Initialize the Vision LLM adapter.

        Args:
            **config: Configuration options:
                - model: str - Model name (claude-opus, nova-pro, nova-lite, claude-sonnet, claude-haiku)
                - model_id: str - Override model ID directly
                - region: str - Region for the configured cloud model
                - prompt: str - Custom extraction prompt
                - infographic_prompt: str - Prompt for infographic-heavy pages
                - max_tokens: int - Max tokens for response
                - temperature: float - Temperature for generation (default: 0)
                - dpi: int - DPI for PDF rendering (default: 150)
                - include_page_markers: bool - Add page number comments (default: True)
                - correct_orientation: bool - Auto-correct rotated pages (default: True)
        """
        self.config = config
        self._client = None

        # Load supported models from config (lazy)
        supported_models = self._get_supported_models()

        # Model configuration
        model_name = config.get("model", "claude-opus")
        model_config = supported_models.get(
            model_name, supported_models.get("claude-opus", {})
        )

        self.model_id = config.get("model_id")
        if self.model_id is None and "model" in config:
            self.model_id = model_config.get("model_id")
        self.max_tokens = config.get("max_tokens", model_config.get("max_tokens", 4096))
        self.temperature = config.get("temperature", 0)

        # Rendering configuration
        self.dpi = config.get("dpi", 300)
        self.include_page_markers = config.get("include_page_markers", True)
        self.correct_orientation = config.get("correct_orientation", True)  # NEW

        # Prompts
        self.extraction_prompt = config.get("prompt", DEFAULT_EXTRACTION_PROMPT)
        self.infographic_prompt = config.get("infographic_prompt", INFOGRAPHIC_PROMPT)

        self.region = config.get("region")

        LOGGER.info(
            "VisionLLMAdapter initialized with model=%s, region=%s, dpi=%d, correct_orientation=%s",
            self.model_id,
            self.region,
            self.dpi,
            self.correct_orientation,
        )

    def _get_llm_adapter(self):
        """Get or create the cloud_lib LLMManager."""
        if self._client is None:
            try:
                from cloud_lib.factory import CloudFactory

                manager_kwargs = {"model_key": "vision"}
                if self.model_id is not None:
                    manager_kwargs["model_id"] = self.model_id
                self._client = CloudFactory.get_llm_manager(**manager_kwargs)
            except Exception as e:
                LOGGER.error("Failed to create LLM adapter: %s", e)
                raise
        return self._client

    def _render_pdf_pages(self, file_path: str) -> List[bytes]:
        """
        Render PDF pages to PNG images using pypdfium2.

        pypdfium2 automatically handles /Rotate metadata during rendering.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError(
                "pypdfium2 is required for PDF rendering. "
                "Install with: pip install pypdfium2"
            )

        images = []
        pdf = pdfium.PdfDocument(file_path)
        scale = self.dpi / 72

        try:
            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # pypdfium2 handles /Rotate automatically
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                images.append(img_bytes)

                LOGGER.debug(
                    "Rendered page %d: %d bytes",
                    page_num + 1,
                    len(img_bytes),
                )
        finally:
            pdf.close()

        return images

    def _render_image(self, file_path: str) -> List[bytes]:
        """Read an image file and return as bytes."""
        with open(file_path, "rb") as f:
            return [f.read()]

    # LLM image payload limit (base64-encoded)
    _MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

    def _ensure_image_under_limit(self, image_bytes: bytes, page_num: int = 0) -> bytes:
        """Compress an image to fit within the LLM payload limit (5 MB)."""
        if len(image_bytes) <= self._MAX_IMAGE_BYTES:
            return image_bytes

        try:
            from PIL import Image
        except ImportError:
            LOGGER.warning(
                "Pillow not installed — cannot compress oversized image "
                "(%d bytes). Sending as-is; LLM may reject it.",
                len(image_bytes),
            )
            return image_bytes

        original_size = len(image_bytes)
        img = Image.open(io.BytesIO(image_bytes))

        # Step 1: re-save as optimised PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= self._MAX_IMAGE_BYTES:
            LOGGER.info(
                "Page %d: optimised PNG reduced %d → %d bytes",
                page_num,
                original_size,
                buf.tell(),
            )
            return buf.getvalue()

        # Step 2: convert to JPEG at decreasing quality
        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        for quality in (90, 80, 70):
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= self._MAX_IMAGE_BYTES:
                LOGGER.info(
                    "Page %d: JPEG q=%d reduced %d → %d bytes",
                    page_num,
                    quality,
                    original_size,
                    buf.tell(),
                )
                return buf.getvalue()

        # Step 3: downscale by 75 % and retry JPEG
        new_w = int(rgb_img.width * 0.75)
        new_h = int(rgb_img.height * 0.75)
        resized = rgb_img.resize((new_w, new_h), Image.LANCZOS)
        for quality in (85, 70):
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= self._MAX_IMAGE_BYTES:
                LOGGER.info(
                    "Page %d: resized (%dx%d) JPEG q=%d reduced %d → %d bytes",
                    page_num,
                    new_w,
                    new_h,
                    quality,
                    original_size,
                    buf.tell(),
                )
                return buf.getvalue()

        # Last resort: return whatever we have (the provider may still reject it)
        LOGGER.warning(
            "Page %d: could not compress below 5 MB (best: %d bytes)",
            page_num,
            buf.tell(),
        )
        return buf.getvalue()

    def _call_vision_model(self, image_bytes: bytes, prompt: str) -> str:
        """Send an image and extraction prompt through the cloud LLM manager."""
        adapter = self._get_llm_adapter()
        image_bytes = self._ensure_image_under_limit(image_bytes)
        try:
            return (
                adapter.converse_vision(
                    model_id=self.model_id,
                    prompt=prompt,
                    image_bytes=image_bytes,
                    image_format=(
                        "jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "png"
                    ),
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                or ""
            )
        except Exception as exc:
            LOGGER.error(
                "Vision inference FAILED | region=%s | model_id=%s | "
                "error_type=%s | error_code=%s | message=%s",
                self.region,
                self.model_id,
                type(exc).__name__,
                getattr(exc, "error_code", ""),
                str(exc)[:500],
            )
            raise

    @staticmethod
    def _detect_infographic(pdf_path: str) -> bool:
        """
        Detect whether a single-page PDF is infographic-heavy.

        Uses pdfplumber text/image coverage analysis.
        Returns ``True`` for infographic pages, ``False`` otherwise.
        """
        try:
            import pdfplumber
        except ImportError:
            return False

        try:
            pdf = pdfplumber.open(pdf_path)
            page = pdf.pages[0]
            page_area = page.width * page.height
            if page_area <= 0:
                pdf.close()
                return False

            words = page.extract_words() or []
            text_area = sum(
                (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
            )
            images = page.images or []
            image_area = sum(
                (img["x1"] - img["x0"]) * (img["bottom"] - img["top"]) for img in images
            )
            pdf.close()

            text_coverage = text_area / page_area
            image_coverage = image_area / page_area
            return image_coverage > 0.3 and text_coverage < 0.3
        except Exception:
            return False

    def _extract_page_content(
        self,
        image_bytes: bytes,
        page_num: int,
        *,
        is_infographic: bool = False,
        prompt_override: Optional[str] = None,
    ) -> str:
        """
        Extract markdown content from a page image.

        Args:
            image_bytes: PNG image bytes
            page_num: Page number (1-indexed)
            is_infographic: If ``True`` the infographic prompt is
                used instead of the default extraction prompt.
            prompt_override: If provided, this prompt is used
                verbatim instead of the extraction/infographic
                prompt selection.

        Returns:
            Extracted markdown text
        """
        if prompt_override:
            prompt = prompt_override
        else:
            prompt = (
                self.infographic_prompt if is_infographic else self.extraction_prompt
            )
        LOGGER.debug(
            "Page %d: using %s prompt",
            page_num,
            "infographic" if is_infographic else "extraction",
        )

        try:
            return self._call_vision_model(image_bytes, prompt)
        except Exception as e:
            LOGGER.error("Failed to extract content from page %d: %s", page_num, e)
            return f"<!-- Error extracting page {page_num}: {e} -->"

    def convert_pages_from_images(
        self,
        page_images: Dict[int, bytes],
        *,
        is_infographic_map: Optional[Dict[int, bool]] = None,
        prompt_override: Optional[str] = None,
    ) -> Dict[int, str]:
        """Convert pre-rendered page images to markdown (no PDF opening).

        This accepts already-rendered PNG bytes so that the caller can
        render once and share the images across multiple consumers,
        eliminating redundant PDF opens.

        Args:
            page_images: Mapping of 1-based page number → PNG bytes.
            is_infographic_map: Per-page infographic flag.  Defaults to
                ``False`` for missing pages.
            prompt_override: If given, used verbatim for every page.

        Returns:
            Dict mapping page number → extracted markdown string.
        """
        is_infographic_map = is_infographic_map or {}
        results: Dict[int, str] = {}

        for page_num in sorted(page_images):
            img_bytes = page_images[page_num]
            is_info = is_infographic_map.get(page_num, False)

            LOGGER.info(
                "VisionLLM processing page %d (infographic=%s)", page_num, is_info
            )
            content = self._extract_page_content(
                img_bytes,
                page_num,
                is_infographic=is_info,
                prompt_override=prompt_override,
            )
            results[page_num] = content.strip() if content else ""

        return results

    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """
        Convert a document to markdown using Vision LLM.

        Args:
            file_path: Path to the document
            mime_type: Optional MIME type
            **kwargs: Additional options:
                - pages: List[int] - Specific pages to convert (1-indexed)
                - prompt_override: str - Use this prompt verbatim for
                  the current call without mutating stored prompts
                - is_infographic: bool - When ``True`` the infographic
                  prompt is used for every page.  When ``False`` the
                  standard extraction prompt is used.  When omitted,
                  each page is auto-detected via text/image coverage.

        Returns:
            Markdown content
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Determine file type and render to images
        if suffix == ".pdf":
            page_images = self._render_pdf_pages(file_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            page_images = self._render_image(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "VisionLLMAdapter supports PDF and image files."
            )

        # Filter specific pages if requested
        pages_to_process = kwargs.get("pages")
        if pages_to_process:
            # Convert to 0-indexed
            indices = [p - 1 for p in pages_to_process if 0 < p <= len(page_images)]
            page_images = [
                (page_images[i], pages_to_process[idx]) for idx, i in enumerate(indices)
            ]
        else:
            page_images = [(img, idx + 1) for idx, img in enumerate(page_images)]

        # Resolve infographic flag:
        #   explicit True/False  → honour the caller's decision
        #   not provided (None)  → auto-detect per page for PDFs
        explicit_infographic = kwargs.get("is_infographic")  # may be None
        auto_detect = explicit_infographic is None and suffix == ".pdf"

        # For auto-detection on single-page PDFs we test the file once
        if auto_detect and len(page_images) == 1:
            auto_result = self._detect_infographic(file_path)
        else:
            auto_result = False  # multi-page auto-detect not yet needed

        # Extract content from each page
        markdown_parts = []
        total_pages = len(page_images)

        for img_bytes, page_num in page_images:
            LOGGER.info("Processing page %d/%d", page_num, total_pages)

            # Add page marker
            if self.include_page_markers:
                markdown_parts.append(f"<!-- page_number: {page_num} -->")

            # Determine infographic flag for this page
            if explicit_infographic is not None:
                is_infographic = explicit_infographic
            elif auto_detect:
                is_infographic = auto_result  # single-page: already computed
            else:
                is_infographic = False

            # Extract content
            content = self._extract_page_content(
                img_bytes,
                page_num,
                is_infographic=is_infographic,
                prompt_override=kwargs.get("prompt_override"),
            )
            markdown_parts.append(content.strip())

        return "\n\n".join(markdown_parts)
