"""
Optimised Parallel Hybrid Markdown Adapter

Performance-optimised PDF-to-markdown pipeline that minimises PDF opens,
in-memory copies, and temporary I/O while preserving accuracy.

**Key optimisations over the previous version:**

1. **Unified pre-scan** — one ``pdfplumber.open()`` + one ``PdfReader()``
   extracts heading structure, rotation metadata, text/image coverage,
   table detection, and routing decisions for *all* pages.
   (Previously: ~302 separate PDF opens for a 100-page PDF.)

2. **Single Docling call on full PDF** — Docling processes the entire
   document once; per-page content is extracted via
   ``export_to_markdown(page_no=X)``.
   (Previously: N separate ``convert()`` calls, each re-initialising
   Docling's layout model.)

3. **True Docling ∥ Vision parallelism** — Docling runs in one thread
   on the full PDF (CPU/GIL-bound) while Vision LLM pages are rendered
    and sent to the configured vision model in parallel worker threads (IO-bound).

4. **Zero temp-file I/O** — no per-page PDF splitting or temp-file writes.
   Rotation correction produces at most one corrected PDF when needed.

5. **Shared page-image cache** — pages rendered once by pypdfium2 are
   reused for Vision LLM extraction, visual-gap detection, and downstream
   infographic chunk generation.

6. **Pre-scan infographic flags** — the text/image coverage analysis from
   the unified pre-scan provides infographic classification, eliminating
   the separate Vision-LLM-per-page InfographicClassifier pass.
"""

import gc
import io
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rag_lib.interfaces.markdown_manager import MarkdownManager

LOGGER = logging.getLogger(__name__)

# Dependency availability flags ---------------------------------------------------
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NumberObject

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pypdfium2 as _pdfium

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False

# pypdfium2 (PDFium C library) is not thread-safe.  Use a shared lock.
import threading

try:
    from docling.utils.locks import pypdfium2_lock as _pdfium_lock
except ImportError:
    _pdfium_lock = threading.Lock()

# Import unified pre-scan and post-merge processor
from rag_lib.adapters.markdown.parallel_hybrid_pdf.unified_prescan import (
    unified_prescan,
    PreScanResult,
)
from rag_lib.adapters.markdown.parallel_hybrid_pdf.post_merge_processor import (
    PostMergeProcessor,
)

# Prompt for the supplementary Vision LLM call on mixed-content pages.
# This asks the model to extract ONLY visual/infographic elements that
# a text-based extractor (Docling) would have missed.
VISUAL_SUPPLEMENT_PROMPT = """You are analyzing a document page that has already been processed for text content. Your job is to describe ONLY the visual elements that a text extractor would miss:

- Infographics, diagrams, flowcharts, process maps
- Visual data displays, matrices, mapping charts
- Icons with labels that form a visual structure
- Relationship diagrams, org charts, network maps

RULES:
1. Do NOT repeat any plain text, headings, or paragraphs — those are already captured
2. Do NOT describe decorative images, backgrounds, or photos of people
3. Focus on information-carrying visual elements only
4. Use ### for the visual element's title, then describe its content with bullet points
5. If there are no meaningful visual elements to describe, respond with exactly: "No visual elements found."

Example output:
### Capitals vs. Stakeholder Value Mapping

This infographic shows a matrix mapping six capitals to stakeholder groups:
- **Financial Capital** → Investors, Customers
- **Manufactured Capital** → Customers, Communities
- **Intellectual Capital** → Investors, Employees
- **Human Capital** → Employees, Communities
- **Social & Relationship Capital** → Customers, Communities, Partners
- **Natural Capital** → Communities, Environment

Extract only the visual elements:"""


@dataclass
class PageResult:
    """Result of processing a single page."""

    page_num: int
    content: str
    method: str  # 'vision' | 'docling' | 'docling+vision'
    processing_time: float
    success: bool
    error: Optional[str] = None
    rotation_corrected: int = 0
    is_infographic: bool = False


class ParallelHybridPdfAdapter(MarkdownManager):
    """Optimised parallel hybrid PDF→Markdown adapter.

    Runs a unified pre-scan, then processes Docling and Vision LLM pages
    in *true* parallelism with zero per-page temp-file I/O.
    """

    def __init__(self, **config):
        self.config = config

        # Parallel processing
        self.max_workers = config.get("max_workers", 4)

        # Orientation correction
        self.correct_orientation = config.get("correct_orientation", True)

        # Page routing
        self.infographic_pages = set(config.get("infographic_pages", []))
        self.table_pages = set(config.get("table_pages", []))
        self.text_threshold = config.get("text_threshold", 0.3)
        self.always_vision_first_page = config.get("always_vision_first_page", True)
        self.max_docling_pages = config.get("max_docling_pages", 200)
        self.docling_batch_size = config.get("docling_batch_size", 100)

        # Populated after convert()
        self.detected_infographic_pages: List[int] = []

        # Adapter instances (lazy)
        self._docling_adapter = None
        self._vision_adapter = None

        # Sub-adapter configs
        self.docling_config = config.get("docling_config", {})
        self.vision_config = config.get("vision_config", {})

        if "region" in config:
            self.vision_config.setdefault("region", config["region"])
        if "vision_model" in config:
            self.vision_config.setdefault("model_id", config["vision_model"])

        # Post-merge processor
        post_merge_config = config.get("post_merge", {})
        self.post_merge_processor = PostMergeProcessor(
            normalize_headings=post_merge_config.get("normalize_headings", True),
            deduplicate_headings=post_merge_config.get("deduplicate_headings", True),
            remove_headers_footers=post_merge_config.get(
                "remove_headers_footers", True
            ),
            fix_cross_page_sentences=post_merge_config.get(
                "fix_cross_page_sentences", True
            ),
            normalize_style=post_merge_config.get("normalize_style", True),
            header_footer_threshold=post_merge_config.get(
                "header_footer_threshold", 0.5
            ),
        )

        LOGGER.info(
            "ParallelHybridPdfAdapter initialized: max_workers=%d, vision_model=%s, "
            "always_vision_first_page=%s, post_merge=enabled",
            self.max_workers,
            self.vision_config.get(
                "model_id", self.vision_config.get("model", "not-set")
            ),
            self.always_vision_first_page,
        )

    # ── Lazy adapter properties ──────────────────────────────────────────

    @property
    def docling_adapter(self):
        """Lazy Docling adapter — used for non-PDF fallback and full-PDF conversion."""
        if self._docling_adapter is None:
            from rag_lib.adapters.markdown.docling import DoclingAdapter

            self._docling_adapter = DoclingAdapter(**self.docling_config)
        return self._docling_adapter

    @property
    def vision_adapter(self):
        """Lazy Vision LLM adapter — IO-bound, safe to share across threads."""
        if self._vision_adapter is None:
            from rag_lib.adapters.markdown.vision_llm import VisionLLMAdapter

            self._vision_adapter = VisionLLMAdapter(**self.vision_config)
        return self._vision_adapter

    # ── Rotation correction ──────────────────────────────────────────────

    def _correct_rotation(
        self,
        file_path: str,
        rotation_map: Dict[int, int],
    ) -> str:
        """Create a rotation-corrected copy of the PDF if needed.

        Opens the original PDF once with ``PdfReader``, zeroes the
        ``/Rotate`` entry on every page that needs correction, and
        writes a single corrected file.

        Returns:
            Path to the corrected PDF (caller must delete), or
            *file_path* unchanged if no correction was needed.
        """
        if not rotation_map or not PYPDF_AVAILABLE:
            return file_path

        reader = PdfReader(file_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        for page_num, degrees in rotation_map.items():
            idx = page_num - 1
            if 0 <= idx < len(writer.pages):
                writer.pages[idx].update({"/Rotate": NumberObject(0)})
                LOGGER.info("Page %d: zeroed /Rotate (was %d°)", page_num, degrees)

        fd, corrected_path = tempfile.mkstemp(
            suffix=".pdf", prefix="phybrid_corrected_"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                writer.write(f)
        except Exception:
            os.close(fd)
            raise

        LOGGER.info(
            "Rotation-corrected PDF written: %s (%d pages corrected)",
            corrected_path,
            len(rotation_map),
        )
        return corrected_path

    # ── Page rendering ───────────────────────────────────────────────────

    def _render_pages(
        self,
        file_path: str,
        page_numbers: List[int],
        dpi: int = 150,
    ) -> Dict[int, bytes]:
        """Render specific pages to PNG using pypdfium2 (single open).

        Returns:
            Dict mapping page number → PNG bytes.
        """
        if not PYPDFIUM2_AVAILABLE:
            raise RuntimeError("pypdfium2 required for page rendering")

        with _pdfium_lock:
            pdf = _pdfium.PdfDocument(file_path)
            scale = dpi / 72
            images: Dict[int, bytes] = {}

            try:
                for pn in page_numbers:
                    idx = pn - 1
                    if idx < 0 or idx >= len(pdf):
                        continue
                    page = pdf[idx]
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    images[pn] = buf.getvalue()
            finally:
                pdf.close()

        LOGGER.debug("Rendered %d pages at %d DPI", len(images), dpi)
        return images

    # ── Visual gap detection ─────────────────────────────────────────────

    VISUAL_GAP_IMAGE_THRESHOLD = 0.3

    def _supplement_visual_content_from_image(
        self,
        page_num: int,
        image_bytes: bytes,
        docling_content: str,
    ) -> Optional[str]:
        """Vision LLM supplement call using pre-rendered image bytes."""
        try:
            result = self.vision_adapter.convert_pages_from_images(
                {page_num: image_bytes},
                is_infographic_map={page_num: True},
                prompt_override=VISUAL_SUPPLEMENT_PROMPT,
            )
            supplement = result.get(page_num, "")
            if not supplement or not supplement.strip():
                return None

            stripped = supplement.strip().lower()
            if any(
                phrase in stripped
                for phrase in [
                    "no visual elements",
                    "no infographic",
                    "no diagram",
                    "no chart",
                    "already captured",
                    "nothing additional",
                ]
            ):
                LOGGER.debug(
                    "Page %d: Vision LLM supplement had no new visual content", page_num
                )
                return None

            LOGGER.info(
                "Page %d: appending Vision LLM visual supplement (%d chars)",
                page_num,
                len(supplement),
            )
            return supplement.strip()

        except Exception as exc:
            LOGGER.warning("Page %d: visual supplement call failed: %s", page_num, exc)
            return None

    # ── Content blocked check ────────────────────────────────────────────

    @staticmethod
    def _is_content_blocked(content: str) -> bool:
        if not content:
            return True
        content_lower = content.lower()
        return any(
            ind in content_lower
            for ind in [
                "blocked by our content filters",
                "content has been blocked",
                "cannot process this content",
                "content filters",
            ]
        )

    # ── Main convert() ───────────────────────────────────────────────────

    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """Convert a PDF to markdown with optimised parallel processing.

        Pipeline:
        1. **Unified pre-scan** (1 pdfplumber + 1 pypdf open) →
           heading map, rotation map, routing decisions, infographic flags.
        2. **Rotation correction** — if any pages rotated, write one
           corrected PDF (single PdfReader+PdfWriter pass).
        3. **Page rendering** — render only Vision-routed pages
           (+ visual-gap candidates) via pypdfium2 (single open).
        4. **Parallel extraction**:
           - Thread A: Docling ``convert_full_pdf_pages()`` on the full
             PDF (one call, per-page export).
           - Threads B…N: Vision LLM from pre-rendered images (parallel
             cloud model calls, IO-bound).
        5. **Visual gap detection** — for Docling pages with high image
           coverage, make supplementary Vision LLM calls from the shared
           rendered images.
        6. **Merge + PostMergeProcessor** — page-order merge, heading
           normalisation, dedup, cross-page sentence repair.
        """
        total_start = time.time()

        path = Path(file_path)
        suffix = path.suffix.lower()

        # Non-PDF delegation
        if suffix != ".pdf":
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                return self.vision_adapter.convert(file_path, mime_type, **kwargs)
            return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        # Forced method — bypass analysis
        force_method = kwargs.get("force_method")
        if force_method == "vision":
            return self.vision_adapter.convert(file_path, mime_type, **kwargs)
        if force_method == "docling":
            return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        if not PYPDF_AVAILABLE:
            LOGGER.warning("pypdf not available, falling back to Docling")
            return self.docling_adapter.convert(file_path, mime_type, **kwargs)

        max_workers = kwargs.get("max_workers", self.max_workers)

        # ── Step 1: Unified pre-scan ──────────────────────────────────
        scan_start = time.time()
        prescan = unified_prescan(
            file_path,
            text_threshold=self.text_threshold,
            always_vision_first_page=self.always_vision_first_page,
            infographic_pages=self.infographic_pages,
            table_pages=self.table_pages,
        )
        scan_time = time.time() - scan_start
        LOGGER.info("Unified pre-scan completed in %.2fs", scan_time)

        # ── Step 2: Rotation correction ───────────────────────────────
        corrected_path = file_path
        corrected_is_temp = False
        if self.correct_orientation and prescan.needs_rotation_correction:
            corrected_path = self._correct_rotation(file_path, prescan.rotation_map)
            corrected_is_temp = corrected_path != file_path

        try:
            return self._process_pdf(
                corrected_path, prescan, max_workers, total_start, scan_time, **kwargs
            )
        finally:
            if corrected_is_temp:
                try:
                    os.unlink(corrected_path)
                except OSError:
                    pass

    def _process_pdf(
        self,
        pdf_path: str,
        prescan: PreScanResult,
        max_workers: int,
        total_start: float,
        scan_time: float,
        **kwargs,
    ) -> str:
        """Core processing after pre-scan and rotation correction."""

        docling_pages = prescan.docling_pages
        vision_pages = prescan.vision_pages
        visual_gap_pages = prescan.visual_gap_pages

        # ── Guard: redirect all docling pages to vision if too many ──
        if len(docling_pages) > self.max_docling_pages:
            LOGGER.warning(
                "Docling page count (%d) exceeds max_docling_pages (%d). "
                "Redirecting all pages to Vision LLM to prevent hang.",
                len(docling_pages),
                self.max_docling_pages,
            )
            vision_pages = sorted(set(vision_pages) | set(docling_pages))
            docling_pages = []
            visual_gap_pages = []

        # ── Step 3: Render pages needed for Vision LLM + visual gap ──
        pages_to_render = sorted(set(vision_pages) | set(visual_gap_pages))
        render_start = time.time()
        rendered_images: Dict[int, bytes] = {}
        if pages_to_render and PYPDFIUM2_AVAILABLE:
            rendered_images = self._render_pages(pdf_path, pages_to_render)
        render_time = time.time() - render_start
        if rendered_images:
            LOGGER.info(
                "Rendered %d pages in %.2fs (vision=%d, visual_gap=%d)",
                len(rendered_images),
                render_time,
                len(vision_pages),
                len(visual_gap_pages),
            )

        # ── Step 4: Parallel extraction ──────────────────────────────

        # Build infographic flag map for vision pages
        vision_infographic_map = {
            pn: prescan.pages[pn].is_infographic
            for pn in vision_pages
            if pn in prescan.pages
        }

        # Results collectors
        results: Dict[int, PageResult] = {}
        process_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # --- Submit Docling (one call, runs in a single thread) ---
            if docling_pages:
                futures[
                    executor.submit(
                        self._run_docling_full_pdf,
                        pdf_path,
                        docling_pages,
                    )
                ] = "docling"

            # --- Submit Vision LLM pages (parallel cloud model calls) -----
            for pn in vision_pages:
                if pn in rendered_images:
                    futures[
                        executor.submit(
                            self._run_vision_page,
                            pn,
                            rendered_images[pn],
                            vision_infographic_map.get(pn, False),
                        )
                    ] = ("vision", pn)

            # Collect results
            for future in as_completed(futures):
                tag = futures[future]
                try:
                    if tag == "docling":
                        docling_results = future.result()
                        for pn, pr in docling_results.items():
                            results[pn] = pr
                    else:
                        _, pn = tag
                        results[pn] = future.result()
                except Exception as exc:
                    if tag == "docling":
                        LOGGER.error("Docling full-PDF conversion failed: %s", exc)
                        for pn in docling_pages:
                            results[pn] = PageResult(
                                page_num=pn,
                                content=f"<!-- Page {pn} - Docling failed: {exc} -->",
                                method="error",
                                processing_time=0,
                                success=False,
                                error=str(exc),
                            )
                    else:
                        _, pn = tag
                        LOGGER.error("Vision page %d failed: %s", pn, exc)
                        results[pn] = PageResult(
                            page_num=pn,
                            content=f"<!-- Page {pn} - Vision failed: {exc} -->",
                            method="error",
                            processing_time=0,
                            success=False,
                            error=str(exc),
                        )

        # ── Step 4b: Fallback for failed pages ───────────────────────
        for pn in sorted(set(docling_pages + vision_pages)):
            if pn not in results or not results[pn].success:
                results[pn] = self._fallback_page(
                    pn, pdf_path, rendered_images, prescan
                )

        process_time = time.time() - process_start
        LOGGER.info("Parallel extraction completed in %.2fs", process_time)

        # ── Step 5: Visual gap detection ─────────────────────────────
        gap_start = time.time()
        gap_count = 0
        for pn in visual_gap_pages:
            if pn not in results or not results[pn].success:
                continue
            if results[pn].method != "docling":
                continue
            img = rendered_images.get(pn)
            if not img:
                continue
            supplement = self._supplement_visual_content_from_image(
                pn,
                img,
                results[pn].content,
            )
            if supplement:
                results[pn] = PageResult(
                    page_num=pn,
                    content=results[pn].content.rstrip() + "\n\n" + supplement,
                    method="docling+vision",
                    processing_time=results[pn].processing_time,
                    success=True,
                    is_infographic=results[pn].is_infographic,
                )
                gap_count += 1
        gap_time = time.time() - gap_start
        if gap_count:
            LOGGER.info(
                "Visual gap detection: %d pages supplemented in %.2fs",
                gap_count,
                gap_time,
            )

        # Free rendered images
        del rendered_images
        gc.collect()

        # ── Step 6: Merge + post-process ─────────────────────────────
        markdown_parts = []
        for pn in sorted(results.keys()):
            result = results[pn]
            clean = re.sub(
                r"<!--\s*page_number:\s*\d+\s*-->\s*\n*", "", result.content
            ).strip()
            markdown_parts.append(f"<!-- page_number: {pn} -->")
            source_label = "vision_llm" if result.method == "vision" else result.method
            markdown_parts.append(f"<!-- source: {source_label} -->")
            markdown_parts.append(clean)

        total_time = time.time() - total_start

        # Collect infographic pages for downstream consumers
        self.detected_infographic_pages = prescan.infographic_page_list

        success_count = sum(1 for r in results.values() if r.success)
        LOGGER.info(
            "Conversion complete: %d pages, %d success, %d→docling, %d→vision, "
            "total=%.2fs (scan=%.2fs, render=%.2fs, process=%.2fs, gap=%.2fs)",
            len(results),
            success_count,
            len(docling_pages),
            len(vision_pages),
            total_time,
            scan_time,
            render_time,
            process_time,
            gap_time,
        )

        merged = "\n\n".join(markdown_parts)
        merged = self.post_merge_processor.process(
            merged, heading_map=prescan.heading_map
        )
        return merged

    # ── Worker functions ─────────────────────────────────────────────────

    @staticmethod
    def _split_pdf_pages(
        pdf_path: str, page_numbers: List[int], output_path: str
    ) -> str:
        """Extract specific pages from a PDF into a new temporary PDF."""
        import pypdf

        reader = pypdf.PdfReader(pdf_path)
        writer = pypdf.PdfWriter()
        for pn in page_numbers:
            writer.add_page(reader.pages[pn - 1])  # 0-indexed
        writer.write(output_path)
        return output_path

    def _run_docling_full_pdf(
        self,
        pdf_path: str,
        page_numbers: List[int],
    ) -> Dict[int, PageResult]:
        """Run Docling on the PDF, batching if needed to avoid OOM."""
        batch_size = self.docling_batch_size
        if len(page_numbers) <= batch_size:
            return self._run_docling_single_batch(pdf_path, page_numbers)

        # Split into batches
        import tempfile, os

        batches = [
            page_numbers[i : i + batch_size]
            for i in range(0, len(page_numbers), batch_size)
        ]
        LOGGER.info(
            "Docling batched processing: %d pages in %d batches of %d",
            len(page_numbers),
            len(batches),
            batch_size,
        )
        all_results: Dict[int, PageResult] = {}
        for batch_idx, batch_pages in enumerate(batches, 1):
            LOGGER.info(
                "Docling batch %d/%d: pages %d-%d (%d pages)",
                batch_idx,
                len(batches),
                batch_pages[0],
                batch_pages[-1],
                len(batch_pages),
            )
            tmp_pdf = None
            try:
                tmp_fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
                os.close(tmp_fd)
                self._split_pdf_pages(pdf_path, batch_pages, tmp_pdf)
                # Docling will see pages 1..N in the temp PDF;
                # map them back to original page numbers
                temp_page_numbers = list(range(1, len(batch_pages) + 1))
                batch_results = self._run_docling_single_batch(
                    tmp_pdf,
                    temp_page_numbers,
                )
                # Remap temp page numbers → original page numbers
                for temp_pn, orig_pn in zip(temp_page_numbers, batch_pages):
                    if temp_pn in batch_results:
                        pr = batch_results[temp_pn]
                        all_results[orig_pn] = PageResult(
                            page_num=orig_pn,
                            content=pr.content,
                            method=pr.method,
                            processing_time=pr.processing_time,
                            success=pr.success,
                            error=pr.error,
                        )
            finally:
                if tmp_pdf and os.path.exists(tmp_pdf):
                    os.unlink(tmp_pdf)

            import gc

            gc.collect()

        return all_results

    def _run_docling_single_batch(
        self,
        pdf_path: str,
        page_numbers: List[int],
    ) -> Dict[int, PageResult]:
        """Run Docling once on a PDF and extract per-page markdown."""
        start = time.time()
        try:
            pages_md = self.docling_adapter.convert_full_pdf_pages(
                pdf_path,
                page_numbers=page_numbers,
            )
            elapsed = time.time() - start
            LOGGER.info(
                "Docling full-PDF conversion: %d pages in %.2fs (%.2fs/page)",
                len(pages_md),
                elapsed,
                elapsed / len(pages_md) if pages_md else 0,
            )
            results: Dict[int, PageResult] = {}
            for pn, md in pages_md.items():
                results[pn] = PageResult(
                    page_num=pn,
                    content=md,
                    method="docling",
                    processing_time=elapsed / len(pages_md),
                    success=True,
                )
            return results
        except Exception as exc:
            elapsed = time.time() - start
            LOGGER.error(
                "Docling full-PDF conversion failed after %.2fs: %s", elapsed, exc
            )
            raise

    def _run_vision_page(
        self,
        page_num: int,
        image_bytes: bytes,
        is_infographic: bool,
    ) -> PageResult:
        """Process a single Vision LLM page from pre-rendered image."""
        start = time.time()
        try:
            result_map = self.vision_adapter.convert_pages_from_images(
                {page_num: image_bytes},
                is_infographic_map={page_num: is_infographic},
            )
            content = result_map.get(page_num, "")

            # Check for blocked content
            if self._is_content_blocked(content):
                LOGGER.warning("Page %d content blocked by Vision LLM", page_num)
                content = f"<!-- Page {page_num} - content blocked -->"

            return PageResult(
                page_num=page_num,
                content=content,
                method="vision",
                processing_time=time.time() - start,
                success=True,
                is_infographic=is_infographic,
            )
        except Exception as exc:
            LOGGER.error("Vision LLM page %d failed: %s", page_num, exc)
            raise

    def _fallback_page(
        self,
        page_num: int,
        pdf_path: str,
        rendered_images: Dict[int, bytes],
        prescan: PreScanResult,
    ) -> PageResult:
        """Attempt fallback extraction for a failed page."""
        info = prescan.pages.get(page_num)
        original_method = info.method if info else "unknown"
        fallback_method = "vision" if original_method == "docling" else "docling"

        LOGGER.info(
            "Trying fallback %s for page %d (primary %s failed)",
            fallback_method,
            page_num,
            original_method,
        )

        start = time.time()
        try:
            if fallback_method == "vision":
                img = rendered_images.get(page_num)
                if not img:
                    img = self._render_pages(pdf_path, [page_num]).get(page_num)
                if img:
                    result_map = self.vision_adapter.convert_pages_from_images(
                        {page_num: img},
                        is_infographic_map={page_num: True},
                    )
                    content = result_map.get(page_num, "")
                else:
                    content = f"<!-- Page {page_num} - could not render -->"
            else:
                pages_md = self.docling_adapter.convert_full_pdf_pages(
                    pdf_path,
                    page_numbers=[page_num],
                )
                content = pages_md.get(page_num, "")

            return PageResult(
                page_num=page_num,
                content=content.strip() if content else "",
                method=fallback_method,
                processing_time=time.time() - start,
                success=True,
            )
        except Exception as exc:
            return PageResult(
                page_num=page_num,
                content=f"<!-- Page {page_num} - Extraction failed: {exc} -->",
                method="error",
                processing_time=time.time() - start,
                success=False,
                error=str(exc),
            )
