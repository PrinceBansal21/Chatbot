import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from rag_lib.interfaces.markdown_manager import MarkdownManager

LOGGER = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        OcrOptions,
        EasyOcrOptions,
        RapidOcrOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    PdfPipelineOptions = None
    OcrOptions = None
    EasyOcrOptions = None
    RapidOcrOptions = None
    TesseractOcrOptions = None
    TesseractCliOcrOptions = None
    InputFormat = None
    PdfFormatOption = None
    DOCLING_AVAILABLE = False

# Try to import macOS-specific OCR (only available on macOS)
try:
    from docling.datamodel.pipeline_options import OcrMacOptions
except ImportError:
    OcrMacOptions = None


class DoclingAdapter(MarkdownManager):
    """
    Adapter for Docling document conversion.
    
    Supports configuration for:
    - OCR settings (engine, languages, enabled, force_full_page)
    - Table extraction (enabled, mode)
    - Image extraction (enabled, resolution)
    - Page markers in output
    
    Supported OCR engines:
    - auto: Auto-select best available engine
    - easyocr: Deep learning OCR (requires easyocr package)
    - rapidocr: Lightweight ONNX-based OCR (requires rapidocr-onnxruntime)
    - tesseract: Tesseract via Python bindings (requires pytesseract)
    - tesseract_cli: Tesseract via command line (requires system tesseract)
    - ocrmac: macOS Vision framework (macOS only)
    """
    
    # Map config engine names to Docling OcrEngine enum values
    OCR_ENGINE_MAP = {
        "auto": "AUTO",
        "easyocr": "EASYOCR",
        "rapidocr": "RAPIDOCR",
        "tesseract": "TESSERACT",
        "tesseract_cli": "TESSERACT_CLI",
        "ocrmac": "OCRMAC",
    }
    
    def __init__(self, **config):
        """
        Initialize the Docling adapter with configuration.
        
        Args:
            **config: Configuration options:
                - ocr: dict with:
                    - enabled: bool (default True)
                    - engine: str - auto|easyocr|rapidocr|tesseract|tesseract_cli|ocrmac
                    - languages: list[str] (default ["en"])
                    - force_full_page: bool - force OCR on entire page (default False)
                - table_extraction: dict with 'enabled', 'mode'
                - image_extraction: dict with 'enabled', 'resolution_scale'
                - page_markers: bool - include page number comments in output (default True)
                - page_marker_strategy: str - DOCX/DOC page marker strategy:
                    * "auto" (default): anchor → layout_aware → docling → xml
                    * "anchor": python-docx anchor injection + LibreOffice→PDF map
                      (most accurate; requires LibreOffice + python-docx + pdfplumber)
                    * "layout_aware": LibreOffice→PDF + pdfplumber fingerprinting only
                    * "docling": Docling's per-page export only
                    * "xml": <w:br w:type="page"/> heuristic only
                    * "none": no markers
        """
        self.config = config
        self._converter = None
        self._page_markers = config.get("page_markers", True)
        self._page_marker_strategy = str(
            config.get("page_marker_strategy", "auto")
        ).lower()
        
        if not DOCLING_AVAILABLE:
            LOGGER.warning("Docling not available. Install with: pip install docling")
            return
            
        try:
            self._converter = self._create_converter()
            LOGGER.info("Docling converter initialized with config: %s", self._summarize_config())
            self._log_device_info()
        except Exception as e:
            LOGGER.warning("Failed to initialize DocumentConverter: %s", e)

    def _log_device_info(self):
        """Log which compute device Docling will use (cpu/cuda)."""
        try:
            from docling.utils.accelerator_utils import decide_device
            resolved = decide_device("auto")
            LOGGER.info("Docling accelerator device: %s", resolved)
        except Exception:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                LOGGER.info("Docling accelerator device (torch): %s", device)
            except ImportError:
                LOGGER.info("Docling accelerator device: unknown (torch not available)")

    def _create_converter(self) -> "DocumentConverter":
        """Create a DocumentConverter with the configured options."""
        if not DOCLING_AVAILABLE:
            return None
            
        # Build pipeline options from config
        pipeline_options = self._build_pipeline_options()
        
        if pipeline_options:
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            return DocumentConverter(format_options=format_options)
        
        # Default converter with no special options
        return DocumentConverter()

    def _build_pipeline_options(self) -> Optional["PdfPipelineOptions"]:
        """Build PdfPipelineOptions from configuration."""
        if not DOCLING_AVAILABLE or not PdfPipelineOptions:
            return None
            
        ocr_cfg = self.config.get("ocr", {})
        table_cfg = self.config.get("table_extraction", {})
        image_cfg = self.config.get("image_extraction", {})
        
        # Only create options if there's something to configure
        if not any([ocr_cfg, table_cfg, image_cfg]):
            return None
            
        options = PdfPipelineOptions()
        
        # Configure OCR
        if ocr_cfg:
            options.do_ocr = ocr_cfg.get("enabled", True)
            
            if options.do_ocr:
                ocr_options = self._build_ocr_options(ocr_cfg)
                if ocr_options:
                    options.ocr_options = ocr_options
        
        # Configure table extraction
        if table_cfg:
            options.do_table_structure = table_cfg.get("enabled", True)
            mode = table_cfg.get("mode", "accurate")
            if mode == "fast":
                # Use faster but less accurate table detection
                try:
                    from docling.datamodel.pipeline_options import TableFormerMode
                    options.table_structure_options = {"mode": TableFormerMode.FAST}
                except ImportError:
                    pass
        
        # Configure image extraction
        if image_cfg:
            options.images_scale = image_cfg.get("resolution_scale", 2.0)
            options.generate_page_images = image_cfg.get("enabled", True)
            # Enable picture image generation by default - this extracts actual images
            # for charts, infographics, and diagrams detected by Docling's layout model
            # Without this, Docling outputs <!-- image --> placeholders without actual image data
            options.generate_picture_images = image_cfg.get("generate_picture_images", True)
        
        return options

    def _build_ocr_options(self, ocr_cfg: Dict[str, Any]) -> Optional[Any]:
        """
        Build OCR options based on the configured engine.
        
        Args:
            ocr_cfg: OCR configuration dict with engine, languages, force_full_page
            
        Returns:
            Engine-specific OCR options object, or None if unavailable
        """
        engine = ocr_cfg.get("engine", "easyocr").lower()
        languages = ocr_cfg.get("languages", ["en"])
        if isinstance(languages, str):
            languages = [languages]
        force_full_page = ocr_cfg.get("force_full_page", False)
        
        # Map engine name to enum
        engine_enum_name = self.OCR_ENGINE_MAP.get(engine, "EASYOCR")
        
        try:
            if engine == "easyocr" and EasyOcrOptions:
                return EasyOcrOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            elif engine == "rapidocr" and RapidOcrOptions:
                return RapidOcrOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            elif engine == "tesseract" and TesseractOcrOptions:
                return TesseractOcrOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            elif engine == "tesseract_cli" and TesseractCliOcrOptions:
                return TesseractCliOcrOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            elif engine == "ocrmac" and OcrMacOptions:
                return OcrMacOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            elif engine == "auto" and OcrOptions:
                # Use base OcrOptions for auto mode
                return OcrOptions(
                    lang=languages,
                    force_full_page_ocr=force_full_page,
                )
            else:
                # Fallback to base OcrOptions
                if OcrOptions:
                    return OcrOptions(
                        lang=languages,
                        force_full_page_ocr=force_full_page,
                    )
        except Exception as e:
            LOGGER.warning("Failed to create OCR options for engine '%s': %s", engine, e)
        
        return None

    def _summarize_config(self) -> str:
        """Return a summary of the active configuration."""
        parts = []
        
        ocr_cfg = self.config.get("ocr", {})
        if ocr_cfg.get("enabled", True):
            engine = ocr_cfg.get("engine", "easyocr")
            langs = ocr_cfg.get("languages", ["en"])
            parts.append(f"OCR({engine}, {langs})")
        else:
            parts.append("OCR(disabled)")
            
        table_cfg = self.config.get("table_extraction", {})
        if table_cfg.get("enabled", True):
            mode = table_cfg.get("mode", "accurate")
            parts.append(f"Tables({mode})")
            
        if self._page_markers:
            parts.append(f"PageMarkers({self._page_marker_strategy})")
            
        return ", ".join(parts) if parts else "defaults"

    # Extensions Docling doesn't recognise natively — copy to a supported
    # extension so the converter accepts them.
    _EXTENSION_ALIASES: dict = {
        ".txt": ".md",
    }

    @staticmethod
    def _find_libreoffice() -> str:
        """Return the path to the LibreOffice CLI or raise."""
        for name in ("soffice", "libreoffice"):
            path = shutil.which(name)
            if path:
                return path
        raise EnvironmentError(
            "LibreOffice CLI not found. Required for .doc → .docx conversion. "
            "Install with: sudo apt-get install -y libreoffice-writer"
        )

    @staticmethod
    def _convert_doc_to_docx(doc_path: str) -> str:
        """Convert a legacy .doc file to .docx using LibreOffice.

        Args:
            doc_path: Path to the .doc file.

        Returns:
            Path to a temporary .docx file (caller must clean up).
        """
        soffice = DoclingAdapter._find_libreoffice()
        output_dir = tempfile.mkdtemp(prefix="doc2docx_")
        cmd = [
            soffice,
            "--headless",
            "--convert-to", "docx",
            "--outdir", output_dir,
            doc_path,
        ]
        LOGGER.info("Converting .doc → .docx: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice .doc → .docx conversion failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )

        docx_name = Path(doc_path).stem + ".docx"
        docx_path = os.path.join(output_dir, docx_name)
        if not os.path.isfile(docx_path):
            raise FileNotFoundError(
                f"Expected .docx not found after conversion: {docx_path}"
            )
        LOGGER.info(
            ".doc → .docx conversion complete: %s (%d bytes)",
            docx_path, os.path.getsize(docx_path),
        )
        return docx_path

    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        # Check if HTML file needs preprocessing
        suffix = Path(file_path).suffix.lower()
        temp_file = None
        doc_converted_path = None
        effective_path = file_path

        # Legacy .doc → .docx conversion via LibreOffice
        if suffix == ".doc":
            try:
                doc_converted_path = self._convert_doc_to_docx(file_path)
                effective_path = doc_converted_path
                suffix = ".docx"
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot process .doc file: LibreOffice conversion to "
                    f".docx failed ({exc}). Ensure libreoffice-writer is "
                    f"installed (apt-get install -y libreoffice-writer)."
                ) from exc

        # Rename unsupported extensions to a Docling-compatible one
        alias = self._EXTENSION_ALIASES.get(suffix)
        if alias:
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb', suffix=alias, delete=False,
            )
            temp_file.write(Path(file_path).read_bytes())
            temp_file.close()
            effective_path = temp_file.name
            LOGGER.debug("Aliased %s → %s for Docling", suffix, alias)

        if suffix in {'.html', '.htm'}:
            # Preprocess HTML to fix srcset issues before docling conversion
            preprocessed_html = self._preprocess_html_for_images(file_path)
            if preprocessed_html:
                # Write to temp file for docling
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.html', 
                    delete=False,
                    encoding='utf-8'
                )
                temp_file.write(preprocessed_html)
                temp_file.close()
                effective_path = temp_file.name
                LOGGER.debug("Preprocessed HTML file for srcset conversion")
        
        try:
            return self._do_convert(effective_path, mime_type, **kwargs)
        finally:
            # Clean up temp file
            if temp_file:
                try:
                    Path(temp_file.name).unlink()
                except Exception:
                    pass
            # Clean up LibreOffice conversion temp directory
            if doc_converted_path:
                try:
                    shutil.rmtree(Path(doc_converted_path).parent, ignore_errors=True)
                except Exception:
                    pass

    def _preprocess_html_for_images(self, file_path: str) -> Optional[str]:
        """
        Preprocess HTML to convert srcset attributes to src for better image handling.
        
        Docling doesn't extract image URLs from srcset attributes, only from src.
        This preprocessing converts srcset to src so images are properly included
        in the markdown output.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Preprocessed HTML content, or None if no preprocessing needed
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            LOGGER.debug("BeautifulSoup not available for HTML preprocessing")
            return None
        
        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            
            modified = False
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src', '').strip()
                srcset = img_tag.get('srcset', '').strip()
                
                # If no src but has srcset, extract URL from srcset
                if not src and srcset:
                    extracted_url = self._extract_url_from_srcset(srcset)
                    if extracted_url:
                        img_tag['src'] = extracted_url
                        LOGGER.debug("Converted srcset to src: %s", extracted_url[:100])
                        modified = True
            
            if modified:
                return str(soup)
            return None
            
        except Exception as e:
            LOGGER.warning("Failed to preprocess HTML for images: %s", e)
            return None

    def _extract_url_from_srcset(self, srcset: str) -> Optional[str]:
        """Extract the best URL from an srcset attribute.

        Delegates to the shared :func:`~rag_lib.adapters.markdown.html_utils.extract_url_from_srcset`.
        """
        from rag_lib.adapters.markdown.html_utils import extract_url_from_srcset
        return extract_url_from_srcset(srcset)

    def convert_full_pdf_pages(
        self,
        file_path: str,
        page_numbers: Optional[list] = None,
    ) -> Dict[int, str]:
        """Convert a full PDF once and return per-page markdown.

        This avoids re-initialising Docling's layout model for each page.
        Docling parses and analyses the entire PDF in a single
        ``DocumentConverter.convert()`` call, then
        ``export_to_markdown(page_no=X)`` extracts cached per-page output.

        Args:
            file_path: Path to the **full** multi-page PDF.
            page_numbers: Optional list of 1-based page numbers to
                extract.  ``None`` returns all pages.

        Returns:
            Dict mapping 1-based page number → cleaned markdown string.
        """
        if not self._converter:
            raise RuntimeError("Docling converter not initialised")

        LOGGER.info("Converting full PDF %s using Docling (single pass)", file_path)
        result = self._converter.convert(Path(file_path))

        pages_map = getattr(result.document, "pages", None) or {}
        if not pages_map or not isinstance(pages_map, dict):
            # Fallback: return full markdown keyed to page 1
            full_md = result.document.export_to_markdown()
            full_md = re.sub(r'<!--\s*page_number:\s*\d+\s*-->\s*\n*', '', full_md).strip()
            return {1: full_md}

        target_pages = page_numbers or sorted(pages_map.keys())
        output: Dict[int, str] = {}

        for page_no in target_pages:
            if page_no not in pages_map:
                continue
            try:
                page_md = result.document.export_to_markdown(page_no=page_no)
            except Exception as exc:
                LOGGER.warning("Docling export_to_markdown(page_no=%d) failed: %s", page_no, exc)
                page_md = ""
            if page_md:
                page_md = re.sub(r'<!--\s*page_number:\s*\d+\s*-->\s*\n*', '', page_md).strip()
            output[page_no] = page_md or ""

        LOGGER.info(
            "Docling full-PDF conversion complete: %d/%d pages extracted",
            len(output), len(target_pages),
        )
        return output

    # Regex for stripping any pre-existing <!-- page_number: N --> markers
    # (Docling sometimes emits incorrect ones, e.g. labelling everything as page 1).
    _PAGE_MARKER_RE = re.compile(r'<!--\s*page_number:\s*\d+\s*-->\s*\n*')

    @classmethod
    def _strip_page_markers(cls, markdown: str) -> str:
        return cls._PAGE_MARKER_RE.sub('', markdown)

    def _apply_docling_page_markers(self, document, full_markdown: str) -> Optional[str]:
        """Inject page markers using Docling's per-page export.

        Returns the augmented markdown, or ``None`` if Docling did not expose
        usable page information.
        """
        pages_map = getattr(document, "pages", None)
        if not pages_map or not isinstance(pages_map, dict):
            return None

        cleaned_full = self._strip_page_markers(full_markdown)
        if len(pages_map) == 1:
            page_no = next(iter(pages_map.keys()))
            return f"<!-- page_number: {page_no} -->\n\n{cleaned_full.strip()}"

        parts: list = []
        for page_no in sorted(pages_map.keys()):
            try:
                page_md = document.export_to_markdown(page_no=page_no)
            except Exception as exc:
                LOGGER.warning(
                    "Docling export_to_markdown(page_no=%d) failed: %s", page_no, exc,
                )
                continue
            if not page_md:
                continue
            page_md_clean = self._strip_page_markers(page_md).strip()
            if page_md_clean:
                parts.append(f"<!-- page_number: {page_no} -->")
                parts.append(page_md_clean)
        return "\n\n".join(parts) if parts else None

    def _apply_docx_page_markers(self, full_markdown: str, docx_path: str) -> str:
        """Apply DOCX page markers using the configured strategy chain.

        For DOCX (and DOC files already converted to DOCX), the priority is:
        layout_aware → docling-pages → xml-heuristic, controlled by
        ``page_marker_strategy`` config. Returns markdown with markers
        injected; if every strategy fails, returns the input unchanged.
        """
        cleaned = self._strip_page_markers(full_markdown)
        strategy = self._page_marker_strategy

        if strategy == "none":
            return cleaned

        order: list
        if strategy == "auto":
            order = ["anchor", "layout_aware", "docling", "xml"]
        elif strategy in ("anchor", "layout_aware", "docling", "xml"):
            order = [strategy]
        else:
            LOGGER.warning(
                "Unknown page_marker_strategy %r; defaulting to auto", strategy,
            )
            order = ["anchor", "layout_aware", "docling", "xml"]

        for name in order:
            if name == "anchor":
                try:
                    from rag_lib.utils.docx_anchor_pages import (
                        inject_anchor_aware_page_markers,
                    )
                    out = inject_anchor_aware_page_markers(
                        docx_path,
                        docling_convert_fn=lambda p: self._converter.convert(
                            Path(p)
                        ).document.export_to_markdown(),
                    )
                    if out is not None:
                        LOGGER.info(
                            "DOCX page markers: anchor strategy succeeded for %s",
                            docx_path,
                        )
                        return out
                    LOGGER.info(
                        "DOCX page markers: anchor strategy unavailable for %s",
                        docx_path,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "DOCX page markers: anchor strategy failed for %s: %s",
                        docx_path, exc,
                    )
            elif name == "layout_aware":
                try:
                    from rag_lib.utils.docx_layout_pages import (
                        inject_layout_aware_page_markers,
                    )
                    out = inject_layout_aware_page_markers(cleaned, docx_path)
                    if out is not None:
                        LOGGER.info(
                            "DOCX page markers: layout_aware succeeded for %s",
                            docx_path,
                        )
                        return out
                    LOGGER.info(
                        "DOCX page markers: layout_aware unavailable for %s",
                        docx_path,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "DOCX page markers: layout_aware failed for %s: %s",
                        docx_path, exc,
                    )
            elif name == "docling":
                # Docling per-page export was already attempted earlier and
                # stored on the instance for this call.
                docling_md = getattr(self, "_last_docling_paged_md", None)
                if docling_md:
                    LOGGER.info(
                        "DOCX page markers: docling per-page export used for %s",
                        docx_path,
                    )
                    return docling_md
                LOGGER.info(
                    "DOCX page markers: docling pages unavailable for %s",
                    docx_path,
                )
            elif name == "xml":
                try:
                    from rag_lib.utils.docx_page_breaks import inject_docx_page_markers
                    out = inject_docx_page_markers(cleaned, docx_path)
                    LOGGER.info(
                        "DOCX page markers: xml heuristic used for %s", docx_path,
                    )
                    return out
                except Exception as exc:
                    LOGGER.warning(
                        "DOCX page markers: xml heuristic failed for %s: %s",
                        docx_path, exc,
                    )

        LOGGER.warning(
            "DOCX page markers: all strategies failed for %s; returning markdown without markers",
            docx_path,
        )
        return cleaned

    def _do_convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """Internal conversion method after any preprocessing."""
        if self._converter:
            try:
                LOGGER.info("Converting %s using Docling", file_path)
                result = self._converter.convert(Path(file_path))

                # Get full markdown from document
                full_markdown = result.document.export_to_markdown()

                if not self._page_markers:
                    return self._strip_page_markers(full_markdown).strip()

                suffix = Path(file_path).suffix.lower()

                # Pre-compute Docling's per-page version once so the DOCX
                # strategy chain can fall back to it without re-exporting.
                docling_paged = self._apply_docling_page_markers(
                    result.document, full_markdown,
                )
                self._last_docling_paged_md = docling_paged

                # DOCX/DOC: layout_aware-first chain (Docling page detection
                # is unreliable on Word documents).
                if suffix in (".docx", ".doc"):
                    try:
                        return self._apply_docx_page_markers(
                            full_markdown, file_path,
                        )
                    finally:
                        self._last_docling_paged_md = None

                # All other formats (PDF, PPTX, etc.): trust Docling's pages.
                if docling_paged is not None:
                    return docling_paged

                return self._strip_page_markers(full_markdown).strip()
            except Exception as exc:
                LOGGER.warning("Docling conversion failed (%s); falling back to raw text", exc)
            finally:
                self._last_docling_paged_md = None
        
        # Fallback logic
        suffix = Path(file_path).suffix.lower()
        if suffix in {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".zip"}:
             raise RuntimeError(f"Cannot convert binary file {file_path} to Markdown without working Docling converter.")

        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
