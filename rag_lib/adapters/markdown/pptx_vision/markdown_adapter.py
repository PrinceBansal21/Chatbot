"""
PPTX Vision Markdown Adapter

Converts PowerPoint (.pptx / .ppt) presentations to structured Markdown
by rendering each slide as an image and sending it to a multimodal Vision
LLM through the configured cloud provider.

Workflow:
    1. If the input is a .ppt file, convert it to .pptx first (via LibreOffice).
    2. Convert the .pptx to PDF using LibreOffice CLI (`soffice --convert-to pdf`).
    3. Render each PDF page as a PNG image (via pypdfium2).
    4. Send each slide image to the Vision LLM with a slide-specific extraction
       prompt that preserves spatial layout, bullet hierarchies, grouped boxes,
       diagrams, and tables.
    5. Assemble per-slide markdown into a single document with page markers.

Prerequisites:
    - LibreOffice CLI (`soffice` or `libreoffice`) must be on PATH.
    - pypdfium2 for PDF→PNG rendering.
    - cloud_lib for cloud model calls.
"""

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from rag_lib.interfaces.markdown_manager import MarkdownManager

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slide-specific extraction prompt
# ---------------------------------------------------------------------------
SLIDE_EXTRACTION_PROMPT = """\
You are a PowerPoint slide analysis expert. Convert this slide image into \
well-structured Markdown that captures **every piece of visible content**.

SLIDE CLASSIFICATION (required, first line of output):
Emit exactly one line before anything else:
`<!-- slide_type: title -->` — if this slide is ONLY a title/section-divider/
    cover slide: a heading plus optional subtitle/byline/date, with NO
    bullets, tables, charts, diagrams, or data.
`<!-- slide_type: content -->` — for every other slide (has bullets, data,
    a table, a diagram, or any substantive body content).

RULES:
1. **Title**: Use `# Slide Title` (H1) ONLY for the main slide title.
2. **Sections / grouped boxes**: Use `## Section Name` for each distinct \
   content group, box, or labelled region on the slide.
3. **Sub-sections**: Use `### Sub-section` and `####` as needed for nested \
   groups.
4. **Bullet points**: Reproduce every bullet exactly. Use nested `-` for \
   sub-bullets.  Pair labels with values on the same line: \
   `- **Label:** value`.
5. **Tables**: Convert tabular content to proper Markdown tables.
6. **Diagrams / flowcharts / circular diagrams**: Describe the structure. \
   List each node/segment as a heading or bullet, preserving parent-child \
   or sequential relationships.
7. **Charts / graphs**: Describe the data with actual values as bullet \
   points under a descriptive heading.
8. **Images / logos**: Mention them briefly (e.g., "Logo: Company Name") \
   but do NOT invent data.
9. **Annotations & footnotes**: Capture notes, disclaimers, fine print, \
   and footer text.
10. Do NOT add commentary or preamble — output raw Markdown only.

SPATIAL POSITIONING (required):
- For EVERY `##` section heading, append its position on the slide in \
  parentheses using one of these labels: \
  `(top-left)`, `(top-center)`, `(top-right)`, \
  `(center-left)`, `(center)`, `(center-right)`, \
  `(bottom-left)`, `(bottom-center)`, `(bottom-right)`, \
  `(left-column)`, `(right-column)`, `(full-width)`, `(spanning)`.
- For the main title `#`, append `(top)` or `(top-center)`.
- For diagrams or images, note their position: e.g., `(center of slide)`.
- Example: `## Safety Monitoring & Engagement (top-left)`

BOX-BOUNDARY ACCURACY (critical):
- Each labelled box or region on the slide has clear visual boundaries \
  (coloured background, border, or header bar).
- Assign every bullet point ONLY to the box it is visually inside. \
  Look carefully at the box edges — adjacent boxes may share a border.
- If a bullet is near the boundary between two boxes, check which \
  header/title bar it falls under, not which box is nearest.
- NEVER move a bullet from one box to another.

Formatting example:
```
# Slide Title (top-center)

## Section A (top-left)
- **Metric 1:** $10.5 bn (+2.3% YoY)
- **Metric 2:** 233,346

## Section B (center-right)
### Sub-group
- Item 1
- Item 2

## Diagram (center)
- Node A → Node B → Node C

> Note: Some items are planned initiatives. (bottom-center)
```

Extract everything visible on this slide using the format above:\
"""

# Prompt variant for slides that are mostly infographic / diagram
INFOGRAPHIC_SLIDE_PROMPT = """\
You are analysing an **infographic or diagram slide** from a PowerPoint \
presentation. Extract ALL visible information into structured Markdown.

SLIDE CLASSIFICATION (required, first line of output):
Emit exactly one line before anything else — for infographic/diagram slides
this will almost always be `<!-- slide_type: content -->`:
`<!-- slide_type: title -->` — ONLY if this slide is actually just a heading/
    section-divider/cover with NO diagram, chart, or data.
`<!-- slide_type: content -->` — otherwise (has a diagram, chart, table, or
    other substantive content).

HEADING HIERARCHY:
- `#`  — Main slide title
- `##` — Each distinct box, region, or labelled group
- `###` / `####` — Nested sub-groups within a box

SPATIAL POSITIONING (required):
- For EVERY `##` section heading, append its position on the slide in \
  parentheses using one of these labels: \
  `(top-left)`, `(top-center)`, `(top-right)`, \
  `(center-left)`, `(center)`, `(center-right)`, \
  `(bottom-left)`, `(bottom-center)`, `(bottom-right)`, \
  `(left-column)`, `(right-column)`, `(full-width)`, `(spanning)`.
- For the main title `#`, append `(top)` or `(top-center)`.
- For diagrams or central visuals, note their position: \
  e.g., `## Diagram (center)`.
- Example: `## IT Infrastructure & Administration (top-right)`

CRITICAL RULES:
1. Every number/value MUST appear on the SAME bullet as its label:
   `- **Label:** value`
2. Include percentages, comparisons, and units in context.
3. For circular / flow diagrams: list each segment as a heading, then \
   its detail bullets.
4. For connected components: describe the connections (e.g., \
   "→", "feeds into", "part of").
5. Group related metrics together under ONE section heading.
6. Capture ALL footnotes, disclaimers, and fine print.
7. Do NOT invent data — only report what is visible.

BOX-BOUNDARY ACCURACY (critical):
- Each labelled box or region has clear visual boundaries \
  (coloured background, border, or header bar).
- Assign every bullet point ONLY to the box it is visually inside. \
  Look carefully at the box edges — adjacent boxes may share a border.
- If a bullet is near the boundary between two boxes, check which \
  header/title bar it falls under, not which box is nearest.
- NEVER move a bullet from one box to another.

Output raw Markdown only — no preamble or commentary.\
"""


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------
class PptxVisionAdapter(MarkdownManager):
    """
    Markdown adapter that converts PPTX/PPT presentations to Markdown
    via slide-image rendering and a multimodal Vision LLM.
    """

    # LLM image payload limit (base64-encoded image)
    _MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

    def __init__(self, **config):
        """
        Args:
            **config:
                model (str): Model alias — nova-pro | nova-lite | claude-sonnet
                    | claude-haiku (default: claude-haiku).
                model_id (str): Override model ID directly.
                region (str): Region for the configured cloud model.
                max_tokens (int): Max tokens for LLM response.
                temperature (float): Generation temperature (default: 0).
                dpi (int): DPI for PDF→PNG rendering (default: 200).
                include_page_markers (bool): Add ``<!-- page_number: N -->``
                    comments (default: True).
                prompt (str): Custom extraction prompt (overrides default).
                infographic_prompt (str): Custom infographic prompt.
                infographic_threshold (float): Image-area / page-area ratio
                    above which the infographic prompt is used (default: 0.30).
        """
        self.config = config
        self._client = None

        # ---- Model ----
        supported = self._get_supported_models()
        model_name = config.get("model", "claude-haiku")

        LOGGER.info(
            "PPTX Vision model resolution: requested_model=%s available_models=%s",
            model_name,
            list(supported.keys()),
        )

        model_cfg = supported.get(model_name)
        if not model_cfg:
            raise ValueError(
                f"PPTX Vision model '{model_name}' not found in configuration. "
                f"Available models: {list(supported.keys())}"
            )

        self.model_id = config.get("model_id")
        if self.model_id is None and "model" in config:
            self.model_id = model_cfg.get("model_id")
        self.max_tokens = config.get("max_tokens", model_cfg.get("max_tokens", 4096))
        self.temperature = config.get("temperature", 0)

        self.region = config.get("region")

        # ---- Rendering ----
        self.dpi = config.get("dpi", 200)
        self.include_page_markers = config.get("include_page_markers", True)

        # ---- Prompts ----
        self.extraction_prompt = config.get("prompt", SLIDE_EXTRACTION_PROMPT)
        self.infographic_prompt = config.get(
            "infographic_prompt", INFOGRAPHIC_SLIDE_PROMPT
        )
        self.infographic_threshold = config.get("infographic_threshold", 0.30)

        LOGGER.info(
            "PptxVisionAdapter initialised: model=%s region=%s dpi=%d",
            self.model_id,
            self.region,
            self.dpi,
        )

    # ------------------------------------------------------------------
    # Model helpers (shared with VisionLLMAdapter logic)
    # ------------------------------------------------------------------
    @classmethod
    def _get_supported_models(cls) -> dict:
        """Load supported vision LLM models from config.

        Falls back to configuration-defined fallback models if primary
        config loading fails. Fallback models are read from config, not
        hardcoded in code.
        """
        try:
            from rag_lib.model_config import get_vision_llm_models

            models = get_vision_llm_models()
            if models:
                return models
        except Exception as exc:
            LOGGER.warning("Failed to load vision_llm_models: %s", exc)

        # Fall back to config-defined models
        try:
            from rag_lib.model_config import get_fallback_vision_llm_models

            fallback = get_fallback_vision_llm_models()
            if fallback:
                LOGGER.info("Using fallback vision LLM models from config")
                return fallback
        except Exception as exc:
            LOGGER.warning("Failed to load fallback vision_llm_models: %s", exc)

        # Last resort: return empty dict (caller will handle missing models)
        LOGGER.error("No vision LLM models available from config or fallback")
        return {}

    # ------------------------------------------------------------------
    # LLM adapter
    # ------------------------------------------------------------------
    def _get_llm_adapter(self):
        if self._client is None:
            from cloud_lib.factory import CloudFactory

            manager_kwargs = {"model_key": "vision"}
            if self.model_id is not None:
                manager_kwargs["model_id"] = self.model_id
            self._client = CloudFactory.get_llm_manager(**manager_kwargs)
        return self._client

    # ------------------------------------------------------------------
    # PPTX → PDF via LibreOffice
    # ------------------------------------------------------------------
    @staticmethod
    def _find_libreoffice() -> str:
        """Return the path to the LibreOffice CLI or raise."""
        for name in ("soffice", "libreoffice"):
            path = shutil.which(name)
            if path:
                return path
        raise EnvironmentError(
            "LibreOffice CLI not found. Install with: "
            "sudo apt-get install -y libreoffice-impress"
        )

    @staticmethod
    def _convert_pptx_to_pdf(pptx_path: str, output_dir: str) -> str:
        """Convert a PPTX (or PPT) file to PDF using LibreOffice.

        Returns the path to the generated PDF.
        """
        soffice = PptxVisionAdapter._find_libreoffice()
        cmd = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            pptx_path,
        ]
        LOGGER.info("Converting PPTX→PDF: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        pdf_name = Path(pptx_path).stem + ".pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(
                f"Expected PDF not found after conversion: {pdf_path}"
            )
        LOGGER.info("PDF created: %s (%d bytes)", pdf_path, os.path.getsize(pdf_path))
        return pdf_path

    # ------------------------------------------------------------------
    # PDF → page images
    # ------------------------------------------------------------------
    def _iter_pdf_pages(self, pdf_path: str) -> Iterator[Tuple[int, bytes]]:
        """Yield PDF pages as ``(page_num, png_bytes)`` using pypdfium2."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError("pypdfium2 required: pip install pypdfium2")

        pdf = pdfium.PdfDocument(pdf_path)
        scale = self.dpi / 72
        try:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                LOGGER.debug(
                    "Rendered page %d: %d bytes (%dx%d)",
                    page_num + 1,
                    len(img_bytes),
                    pil_image.width,
                    pil_image.height,
                )
                yield page_num + 1, img_bytes
        finally:
            pdf.close()

    # ------------------------------------------------------------------
    # Image compression (stay under LLM 5 MB payload limit)
    # ------------------------------------------------------------------
    def _ensure_image_under_limit(self, image_bytes: bytes, page_num: int = 0) -> bytes:
        if len(image_bytes) <= self._MAX_IMAGE_BYTES:
            return image_bytes
        try:
            from PIL import Image
        except ImportError:
            LOGGER.warning("Pillow not installed — cannot compress oversized image.")
            return image_bytes

        img = Image.open(io.BytesIO(image_bytes))

        # Try optimised PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= self._MAX_IMAGE_BYTES:
            return buf.getvalue()

        # JPEG at decreasing quality
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        for q in (90, 80, 70):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q, optimize=True)
            if buf.tell() <= self._MAX_IMAGE_BYTES:
                return buf.getvalue()

        # Down-scale 75% + JPEG
        new_w, new_h = int(rgb.width * 0.75), int(rgb.height * 0.75)
        resized = rgb.resize((new_w, new_h), Image.LANCZOS)
        for q in (85, 70):
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=q, optimize=True)
            if buf.tell() <= self._MAX_IMAGE_BYTES:
                return buf.getvalue()

        LOGGER.warning("Page %d: could not compress below 5 MB", page_num)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # PPTX shape-level text extraction (supplements vision)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pptx_slide_text(pptx_path: str, slide_index: int) -> str:
        """Extract shape-grouped text from a PPTX slide using python-pptx.

        Unlike pdfplumber (which gives a flat, spatially-scattered text
        dump), python-pptx reads the actual XML shapes — preserving the
        exact grouping of bullets within their parent box.  This ensures
        that boundary-adjacent items (e.g. "Radiotherapy QA Central
        Review" in the Electronic Data Capture box) are never
        misattributed to a neighbouring section.

        Returns a formatted string like:
            [Box Title]\n- bullet 1\n- bullet 2\n\n[Box Title 2]\n...
        """
        try:
            from pptx import Presentation
        except ImportError:
            return ""
        try:
            prs = Presentation(pptx_path)
            if slide_index >= len(prs.slides):
                return ""
            slide = prs.slides[slide_index]
            return PptxVisionAdapter._shapes_to_grouped_text(slide.shapes)
        except Exception as exc:
            LOGGER.debug(
                "python-pptx extraction failed for slide %d: %s", slide_index + 1, exc
            )
            return ""

    @staticmethod
    def _shapes_to_grouped_text(shapes, depth: int = 0) -> str:
        """Recursively extract text from shapes, keeping each shape's
        paragraphs together as a group."""
        groups: List[str] = []
        for shape in shapes:
            # Recurse into group shapes
            if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
                inner = PptxVisionAdapter._shapes_to_grouped_text(
                    shape.shapes, depth + 1
                )
                if inner.strip():
                    groups.append(inner)
                continue

            if not hasattr(shape, "text_frame") or not shape.has_text_frame:
                continue

            paragraphs = []
            for para in shape.text_frame.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            if not paragraphs:
                continue

            # First paragraph is typically the box heading;
            # remaining paragraphs are bullets.
            if len(paragraphs) == 1:
                groups.append(paragraphs[0])
            else:
                header = paragraphs[0]
                bullets = "\n".join(f"- {p}" for p in paragraphs[1:])
                groups.append(f"[{header}]\n{bullets}")

        return "\n\n".join(groups)

    @staticmethod
    def _extract_pdf_text(pdf_path: str, page_index: int) -> str:
        """Fallback: extract flat text from a PDF page via pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            return ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index >= len(pdf.pages):
                    return ""
                return (pdf.pages[page_index].extract_text() or "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Infographic auto-detection
    # ------------------------------------------------------------------
    def _detect_infographic(self, pdf_path: str, page_index: int) -> bool:
        """Return True if the given page is infographic-heavy."""
        try:
            import pdfplumber
        except ImportError:
            return False
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index >= len(pdf.pages):
                    return False
                page = pdf.pages[page_index]
                area = page.width * page.height
                if area <= 0:
                    return False
                words = page.extract_words() or []
                text_area = sum(
                    (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
                )
                imgs = page.images or []
                img_area = sum(
                    (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) for im in imgs
                )
                text_cov = text_area / area
                img_cov = img_area / area
                return img_cov > self.infographic_threshold and text_cov < 0.30
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Slide-type classification marker (title vs content)
    # ------------------------------------------------------------------
    _SLIDE_TYPE_RE = re.compile(
        r"^\s*<!--\s*slide_type:\s*(title|content)\s*-->\s*\n?", re.IGNORECASE
    )

    @classmethod
    def _extract_and_strip_slide_type(cls, content: str) -> Tuple[str, str]:
        """Pull the ``<!-- slide_type: title|content -->`` marker off the top
        of the LLM response. Returns ``(slide_type, remaining_content)``.
        Defaults to ``"content"`` if the model omitted the marker.
        """
        if not content:
            return "content", content
        m = cls._SLIDE_TYPE_RE.match(content)
        if not m:
            return "content", content
        return m.group(1).lower(), content[m.end() :].lstrip()

    # ------------------------------------------------------------------
    # Vision LLM call (Claude / Nova)
    # ------------------------------------------------------------------
    def _call_vision_llm(self, image_bytes: bytes, prompt: str) -> str:
        """Send an image + prompt to the configured vision LLM."""
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
                "adapter=pptx_vision | error_type=%s | error_code=%s | message=%s",
                self.region,
                self.model_id,
                type(exc).__name__,
                getattr(exc, "error_code", ""),
                str(exc)[:500],
            )
            raise

    # ------------------------------------------------------------------
    # convert() — MarkdownManager interface
    # ------------------------------------------------------------------
    @staticmethod
    def _build_text_hint(grouped_text: str) -> str:
        """Build a prompt supplement from shape-grouped PPTX text."""
        if not grouped_text:
            return ""
        # Truncate to avoid blowing up the prompt
        if len(grouped_text) > 4000:
            grouped_text = grouped_text[:4000] + "…"
        return (
            "\n\n---\n"
            "IMPORTANT — The following text was extracted directly from the "
            "PowerPoint slide shapes.  Each [Header] followed by bullets "
            "represents one box/region on the slide.  The bullets listed under "
            "each header belong EXCLUSIVELY to that box — do NOT move bullets "
            "between sections.  Use this as the authoritative source for "
            "which bullet belongs to which section.  Also use it to capture "
            "ALL abbreviations, acronyms, and fine-print content that may be "
            "hard to read in the image:\n\n"
            f"{grouped_text}\n---"
        )

    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """
        Convert a PPTX/PPT file to Markdown.

        Args:
            file_path: Path to the .pptx or .ppt file.
            mime_type: Optional MIME type (unused — file extension is used).
            **kwargs:
                pages (List[int]): 1-based page numbers to process.
                is_infographic (bool | None): Force infographic prompt
                    for all slides. ``None`` → auto-detect per slide.
                prompt_override (str): Use this prompt verbatim.
                slide_image_callback: Optional callable(page_num: int,
                    png_bytes: bytes, slide_markdown: str) invoked for
                    each rendered slide so callers can persist the images.

        Returns:
            Combined Markdown for all slides.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in {".pptx", ".ppt"}:
            raise ValueError(
                f"PptxVisionAdapter only supports .pptx/.ppt files, got: {suffix}"
            )

        with tempfile.TemporaryDirectory(prefix="pptx_vision_") as tmp_dir:
            # Step 1: Convert PPTX → PDF
            pdf_path = self._convert_pptx_to_pdf(str(path), tmp_dir)

            # Step 2: Configure optional page filtering
            pages_filter = kwargs.get("pages")
            allowed_pages = set(pages_filter) if pages_filter else None

            # Infographic flag
            explicit_infographic = kwargs.get("is_infographic")
            prompt_override = kwargs.get("prompt_override")
            slide_image_callback = kwargs.get("slide_image_callback")

            # Step 3: Extract Markdown from each slide
            markdown_parts: List[str] = []
            processed = 0
            for page_num, img_bytes in self._iter_pdf_pages(pdf_path):
                if allowed_pages is not None and page_num not in allowed_pages:
                    continue

                processed += 1
                LOGGER.info("Processing slide %d", page_num)

                if self.include_page_markers:
                    markdown_parts.append(f"<!-- page_number: {page_num} -->")

                slide_type = "content"

                # Extract shape-grouped text from PPTX (preferred)
                # Falls back to PDF text layer if python-pptx fails
                slide_text = self._extract_pptx_slide_text(str(path), page_num - 1)
                if not slide_text:
                    slide_text = self._extract_pdf_text(pdf_path, page_num - 1)
                text_hint = self._build_text_hint(slide_text)

                # Determine prompt
                if prompt_override:
                    prompt = prompt_override
                elif explicit_infographic is True:
                    prompt = self.infographic_prompt + text_hint
                elif explicit_infographic is False:
                    prompt = self.extraction_prompt + text_hint
                else:
                    # Auto-detect
                    is_info = self._detect_infographic(pdf_path, page_num - 1)
                    base_prompt = (
                        self.infographic_prompt if is_info else self.extraction_prompt
                    )
                    prompt = base_prompt + text_hint

                try:
                    raw_content = self._call_vision_llm(img_bytes, prompt)
                    slide_type, content = self._extract_and_strip_slide_type(
                        raw_content or ""
                    )
                    if self.include_page_markers:
                        markdown_parts.append(f"<!-- slide_type: {slide_type} -->")
                    markdown_parts.append(content.strip() if content else "")
                except Exception as e:
                    LOGGER.error("Failed on slide %d: %s", page_num, e)
                    markdown_parts.append(
                        f"<!-- Error extracting slide {page_num}: {e} -->"
                    )
                    content = None

                # Notify caller with rendered slide image so it can be persisted
                if slide_image_callback:
                    try:
                        slide_image_callback(
                            page_num, img_bytes, (content or "").strip(), slide_type
                        )
                    except Exception as cb_err:
                        LOGGER.warning(
                            "slide_image_callback failed for slide %d: %s",
                            page_num,
                            cb_err,
                        )

            LOGGER.info(
                "Rendered and processed %d slide images from %s", processed, path.name
            )

        return "\n\n".join(markdown_parts)
