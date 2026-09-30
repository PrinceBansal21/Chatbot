"""Vision LLM-based infographic page classifier.

Classifies each page of a PDF as infographic or non-infographic using a
multimodal Vision LLM (Claude Haiku by default) via the configured LLM Converse API.

Pages are classified **in parallel** using a :class:`ThreadPoolExecutor` so
classification can run concurrently with other pipeline stages (e.g. chunking)
without adding sequential latency.

Usage::

    classifier = InfographicClassifier(region="us-east-1")
    results = classifier.classify_pdf("/tmp/report.pdf")
    # results == {1: False, 2: True, 3: False, ...}
"""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)
_log = logging.getLogger(__name__)  # standard logger for debug/info

# ── Prompt ────────────────────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = (
    "Look at this document page image. Determine whether it is an INFOGRAPHIC page.\n\n"
    "An infographic page is one that primarily communicates information through "
    "visual elements such as:\n"
    "- Charts, graphs, diagrams, or flowcharts\n"
    "- Illustrated timelines or process flows\n"
    "- Visual data representations (pie charts, bar graphs, maps)\n"
    "- Pages dominated by images, icons, or visual design elements with minimal body text\n"
    "- Marketing/brochure-style pages with heavy visual layout\n\n"
    "A page is NOT an infographic if it is:\n"
    "- Primarily text-based (paragraphs, bullet points, body text)\n"
    "- A standard table (rows and columns of text/numbers)\n"
    "- A cover page with just a title and logo\n"
    "- A table of contents or index\n\n"
    "Respond with ONLY one word: INFOGRAPHIC or TEXT"
)

# ── Image size limits ─────────────────────────────────────────────────────────
# LLM Converse API measures the base64-encoded size and enforces a 5 MiB
# Base64 encoding inflates raw bytes by ~4/3×, so raw bytes must be
# kept under ~3.75 MB to stay safely within the 5 MiB encoded limit.
_MAX_IMAGE_RAW_BYTES = 3_750_000


class InfographicClassifier:
    """Classify PDF pages as infographic or non-infographic using Vision LLM.

    Parameters
    ----------
    region : str or None
        Deprecated compatibility argument. Cloud-lib resolves provider location.
    model_id : str or None
        Optional concrete model override. Cloud-lib otherwise resolves ``vision``.
    max_workers : int
        Maximum concurrent Vision LLM calls for page classification.
    dpi : int
        Resolution used to render PDF pages to images.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        max_workers: int = 5,
        dpi: int = 150,
    ) -> None:
        self.region = region
        self.model_id = model_id
        self.max_workers = max_workers
        self.dpi = dpi
        self._client: Optional[Any] = None  # lazy Converse LLM client

    # ── lazy client ──────────────────────────────────────────────────────────

    @property
    def client(self) -> Any:
        """Lazy-init :class:`LLMConverseClient`."""
        if self._client is None:
            from services.llm.llm_converse_client import LLMConverseClient
            self._client = LLMConverseClient(model_key="vision")
        return self._client

    # ── public API ───────────────────────────────────────────────────────────

    def classify_pdf(
        self,
        file_path: str,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, bool]:
        """Classify every (or selected) page of a PDF.

        Parameters
        ----------
        file_path : str
            Path to a PDF file on the local file system.
        page_numbers : list of int or None
            1-based page numbers to classify.  *None* means all pages.

        Returns
        -------
        dict
            Mapping ``{page_number: is_infographic}`` for each requested page.
        """
        start = time.time()

        page_images = self._render_pages(file_path, page_numbers)
        if not page_images:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                "InfographicClassifier: no pages to classify",
            )
            return {}

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"InfographicClassifier: classifying {len(page_images)} pages "
            f"with {self.max_workers} workers (model={self.model_id})",
        )

        results: Dict[int, bool] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_page = {
                pool.submit(self._classify_page, page_num, img_bytes): page_num
                for page_num, img_bytes in page_images.items()
            }
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    results[page_num] = future.result()
                except Exception as exc:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_CHUNK,
                        f"InfographicClassifier: page {page_num} classification "
                        f"failed — defaulting to False: {exc}",
                    )
                    results[page_num] = False

        elapsed = time.time() - start
        infographic_pages = sorted(p for p, v in results.items() if v)

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"InfographicClassifier: done in {elapsed:.1f}s — "
            f"{len(infographic_pages)}/{len(results)} infographic pages "
            f"({infographic_pages or 'none'})",
        )

        return results

    # ── internals ────────────────────────────────────────────────────────────

    def _render_pages(
        self,
        file_path: str,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, bytes]:
        """Render PDF pages to PNG images.

        Returns a dict mapping 1-based page numbers to PNG bytes.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            LOGGER.log_warning(
                ProcessLogger.STAGE_CHUNK,
                "InfographicClassifier: pypdfium2 not installed — skipping classification",
            )
            return {}

        pdf = pdfium.PdfDocument(file_path)
        total = len(pdf)

        if page_numbers is None:
            page_numbers = list(range(1, total + 1))
        else:
            page_numbers = [p for p in page_numbers if 1 <= p <= total]

        scale = self.dpi / 72
        images: Dict[int, bytes] = {}

        for page_num in page_numbers:
            page = pdf[page_num - 1]  # 0-indexed
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            img_bytes = self._ensure_under_limit(img_bytes, page_num)
            images[page_num] = img_bytes

        pdf.close()
        return images

    def _classify_page(self, page_num: int, image_bytes: bytes) -> bool:
        """Classify a single page image via Vision LLM.

        Returns ``True`` if the page is classified as an infographic.
        """
        # Detect format from magic bytes
        img_format = "jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "png"

        response_text = self.client.converse(
            model_id=self.model_id,
            prompt=_CLASSIFICATION_PROMPT,
            image_bytes=image_bytes,
            image_format=img_format,
            max_tokens=10,
            temperature=0.0,
        )

        if response_text is None:
            _log.debug("Page %d: empty response — treating as non-infographic", page_num)
            return False

        answer = response_text.strip().upper()
        is_infographic = "INFOGRAPHIC" in answer

        _log.debug(
            "Page %d: classified as %s (raw=%r)",
            page_num,
            "INFOGRAPHIC" if is_infographic else "TEXT",
            answer,
        )
        return is_infographic

    # ── image compression ────────────────────────────────────────────────────

    @staticmethod
    def _ensure_under_limit(image_bytes: bytes, page_num: int = 0) -> bytes:
        """Ensure image is under the LLM raw-bytes limit (~3.75 MB).

        Progressive strategy:
        1. Re-encode as optimised PNG.
        2. Convert to JPEG at quality 90 → 80 → 70.
        3. Downscale by 75% and retry JPEG.
        """
        if len(image_bytes) <= _MAX_IMAGE_RAW_BYTES:
            return image_bytes

        try:
            from PIL import Image
        except ImportError:
            LOGGER.log_warning(
                ProcessLogger.STAGE_CHUNK,
                f"Pillow not installed — cannot compress page {page_num} image ({len(image_bytes)} bytes)",
            )
            return image_bytes

        original_size = len(image_bytes)
        img = Image.open(io.BytesIO(image_bytes))

        # Step 1: optimised PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= _MAX_IMAGE_RAW_BYTES:
            _log.info(
                "Page %d: optimised PNG %d → %d bytes",
                page_num, original_size, buf.tell(),
            )
            return buf.getvalue()

        # Step 2: JPEG at decreasing quality
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        for quality in (90, 80, 70):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= _MAX_IMAGE_RAW_BYTES:
                _log.info(
                    "Page %d: JPEG q=%d reduced %d → %d bytes",
                    page_num, quality, original_size, buf.tell(),
                )
                return buf.getvalue()

        # Step 3: downscale + JPEG
        new_w = int(rgb.width * 0.75)
        new_h = int(rgb.height * 0.75)
        resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
        for quality in (85, 70):
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= _MAX_IMAGE_RAW_BYTES:
                _log.info(
                    "Page %d: resized JPEG q=%d reduced %d → %d bytes",
                    page_num, quality, original_size, buf.tell(),
                )
                return buf.getvalue()

        LOGGER.log_warning(
            ProcessLogger.STAGE_CHUNK,
            f"Page {page_num}: could not compress below "
            f"{_MAX_IMAGE_RAW_BYTES / 1_000_000:.1f} MB (best: {buf.tell()} bytes)",
        )
        return buf.getvalue()
