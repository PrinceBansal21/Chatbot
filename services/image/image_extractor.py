"""Image extraction and AI description service.

Extracts images from PDF / HTML documents, generates AI descriptions via
a Vision LLM, and uploads the results to cloud storage.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import ProcessLogger, get_logger
from services.llm import LLMConverseClient

LOGGER = get_logger(__name__)


class ImageProcessor:
    """
    Service for extracting, describing, and embedding images in documents.

    Responsibilities:
    - Extract images from PDF and HTML documents
    - Generate AI descriptions using a Vision LLM
    - Store images to object storage via ImageRepository
    - Embed image URLs into markdown content
    """

    def __init__(
        self,
        image_repository,
        config: Dict[str, Any],
        region: Optional[str] = None,
    ):
        """
        Initialize the ImageProcessor.

        Args:
            image_repository: ImageRepository instance for object storage
            config: Image extraction configuration dict
            region: Deprecated compatibility argument; cloud-lib resolves location.
        """
        self.image_repository = image_repository
        self.config = config
        self._converse = LLMConverseClient(model_key="vision")

    def extract_and_store_images(
        self,
        file_path: str,
        file_id: str,
        metadata: Dict[str, str],
        asset_root: Optional[str] = None,
        payload_image_cfg: Optional[Dict[str, Any]] = None,
        document_prefix: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract images from a document and store them to cloud storage.

        Supports both PDF and HTML files:
        - PDF: Extracts embedded images using pypdfium2
        - HTML: Extracts images from <img> tags

        Args:
            file_path: Path to the document file
            file_id: Unique file identifier
            metadata: Base metadata dictionary
            asset_root: Root directory for resolving relative image paths (HTML only)
            payload_image_cfg: Per-payload image extraction config (overrides global config)
            document_prefix: Optional object storage key prefix of the original document.
                When provided, images are stored alongside the document.

        Returns:
            Image manifest dictionary or None if extraction fails
        """
        try:
            from rag_lib.adapters.markdown.image_aware import ImageExtractor, ImageManifest

            LOGGER.log_stage(ProcessLogger.STAGE_STORAGE, f"Extracting images from: {file_path}")

            # Merge payload config with global config (payload takes precedence)
            image_cfg = self.config.copy()
            if payload_image_cfg:
                image_cfg.update(payload_image_cfg)

            ext = Path(file_path).suffix.lower()
            images = self._extract_images_by_type(file_path, file_id, ext, image_cfg, asset_root)

            if not images:
                LOGGER.log_stage(ProcessLogger.STAGE_STORAGE, "No images found in document")
                return None

            LOGGER.log_stage(ProcessLogger.STAGE_STORAGE, f"Extracted {len(images)} images")

            # Generate descriptions if enabled
            if image_cfg.get("enable_descriptions", True):
                # Resolve description model strictly from config \u2014 no hardcoded default.
                description_model = image_cfg.get("description_model")
                classify_infographics = image_cfg.get("classify_infographics", True)
                LOGGER.log_stage(
                    ProcessLogger.STAGE_STORAGE,
                    f"Generating AI descriptions for {len(images)} images using {description_model}"
                    f"{' (with infographic classification)' if classify_infographics else ''}",
                )
                images = self.generate_descriptions(
                    images, description_model,
                    classify_infographics=classify_infographics,
                )

            # Filter out decorative/irrelevant images before upload
            # Keep: infographics, meaningful images (not decorative)
            # Drop: logos, navigation bars, icons, banners, watermarks, etc.
            if image_cfg.get("filter_decorative_images", True):
                before_count = len(images)
                images = [
                    img for img in images
                    if getattr(img, 'is_infographic', False)
                    or not getattr(img, 'is_decorative', False)
                ]
                filtered_count = before_count - len(images)
                if filtered_count > 0:
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_STORAGE,
                        f"Filtered {filtered_count} decorative/irrelevant images "
                        f"(keeping {len(images)} meaningful images)",
                    )

                if not images:
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_STORAGE,
                        "All images were decorative — no images to store",
                    )
                    return None

            # Upload images to object storage
            LOGGER.log_stage(ProcessLogger.STAGE_STORAGE, f"Uploading {len(images)} images to cloud storage")
            stored_images = self._upload_images(images, file_id, document_prefix=document_prefix)

            # Create and store manifest
            manifest = ImageManifest(
                document_id=file_id,
                source_file=file_path,
                images=stored_images,
                total_pages=self._get_page_count(file_path),
                extraction_timestamp=datetime.utcnow().isoformat(),
            )

            manifest_dict = manifest.to_dict()
            manifest_location = self.image_repository.put_manifest(
                file_id, manifest_dict, document_prefix=document_prefix,
            )
            manifest_dict["manifest_location"] = manifest_location

            LOGGER.log_stage(
                ProcessLogger.STAGE_STORAGE,
                f"Stored {len(stored_images)} images and manifest for {file_id}",
            )

            return manifest_dict

        except ImportError as e:
            LOGGER.log_warning(ProcessLogger.STAGE_STORAGE, f"Image extraction not available: {e}")
            return None
        except Exception as e:
            LOGGER.log_warning(ProcessLogger.STAGE_STORAGE, f"Image extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    #  Slide image persistence (PPTX)
    # ------------------------------------------------------------------

    def store_slide_images(
        self,
        file_id: str,
        slide_images: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Upload rendered PPTX slide images to cloud storage and return an image manifest.

        Each entry in *slide_images* is a dict with:
            - page_number (int): 1-based slide number
            - image_bytes (bytes): PNG bytes of the rendered slide
            - description (str): Markdown extracted by the Vision LLM

        Returns:
            Image manifest dict (same shape as extract_and_store_images) or None.
        """
        if not slide_images:
            return None

        try:
            stored = []
            for slide in slide_images:
                page_num = slide["page_number"]
                image_id = f"slide-{page_num:03d}"
                try:
                    storage_uri = self.image_repository.put_image(
                        document_id=file_id,
                        image_id=image_id,
                        image_bytes=slide["image_bytes"],
                        image_format="png",
                    )
                    stored.append({
                        "image_id": image_id,
                        "storage_path": storage_uri,
                        "page": str(page_num),
                        "description": (slide.get("description") or "")[:500],
                        "format": "png",
                        "image_type": "slide",
                    })
                except Exception as e:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_STORAGE,
                        f"Failed to upload slide image {image_id}: {e}",
                    )

            if not stored:
                return None

            manifest_dict = {
                "document_id": file_id,
                "source_file": f"{file_id} (slide renders)",
                "total_pages": len(slide_images),
                "image_count": len(stored),
                "images": stored,
            }
            LOGGER.log_stage(
                ProcessLogger.STAGE_STORAGE,
                f"Stored {len(stored)} slide images for {file_id}",
            )
            return manifest_dict

        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_STORAGE,
                f"Slide image storage failed: {e}",
            )
            return None

    # ------------------------------------------------------------------
    #  Extension → extractor registry
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_extractor(file_path, file_id, image_cfg, _asset_root):
        from rag_lib.adapters.markdown.image_aware import ImageExtractor

        return ImageExtractor(
            min_width=image_cfg.get("min_width", 50),
            min_height=image_cfg.get("min_height", 50),
            max_area_ratio=image_cfg.get("max_area_ratio", 0.85),
            extract_full_pages=image_cfg.get("extract_full_pages", False),
        ).extract_images(file_path, document_id=file_id)

    @staticmethod
    def _html_extractor(file_path, file_id, image_cfg, asset_root):
        from rag_lib.adapters.markdown.image_aware import HtmlImageExtractor

        return HtmlImageExtractor(
            min_width=image_cfg.get("min_width", 50),
            min_height=image_cfg.get("min_height", 50),
            fetch_remote=image_cfg.get("fetch_remote_images", False),
            convert_svg_to_png=image_cfg.get("convert_svg_to_png", True),
        ).extract_images(
            html_path=file_path,
            asset_root=asset_root or str(Path(file_path).parent),
            document_id=file_id,
        )

    @staticmethod
    def _docx_extractor(file_path, file_id, image_cfg, _asset_root):
        from rag_lib.adapters.markdown.image_aware import DocxImageExtractor

        return DocxImageExtractor(
            min_width=image_cfg.get("min_width", 50),
            min_height=image_cfg.get("min_height", 50),
        ).extract_images(file_path, document_id=file_id)

    @staticmethod
    def _pptx_extractor(file_path, file_id, image_cfg, _asset_root):
        from rag_lib.adapters.markdown.image_aware import PptxImageExtractor

        return PptxImageExtractor(
            min_width=image_cfg.get("min_width", 50),
            min_height=image_cfg.get("min_height", 50),
        ).extract_images(file_path, document_id=file_id)

    # Maps file extension → extraction callable
    _EXTRACTOR_REGISTRY = {
        ".pdf": _pdf_extractor.__func__,
        ".html": _html_extractor.__func__,
        ".htm": _html_extractor.__func__,
        ".docx": _docx_extractor.__func__,
        ".doc": _docx_extractor.__func__,
        ".pptx": _pptx_extractor.__func__,
        ".ppt": _pptx_extractor.__func__,
    }

    def _extract_images_by_type(
        self,
        file_path: str,
        file_id: str,
        ext: str,
        image_cfg: Dict[str, Any],
        asset_root: Optional[str] = None,
    ) -> List:
        """Extract images based on file type using the extractor registry."""
        extractor_fn = self._EXTRACTOR_REGISTRY.get(ext)
        if extractor_fn is None:
            LOGGER.log_stage(
                ProcessLogger.STAGE_STORAGE,
                f"Image extraction not supported for file type: {ext}",
            )
            return []

        try:
            return extractor_fn(file_path, file_id, image_cfg, asset_root)
        except ImportError as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_STORAGE,
                f"Image extraction not available for {ext}: {e}",
            )
            return []
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_STORAGE,
                f"Image extraction failed for {ext}: {e}",
            )
            return []

    def _upload_images(self, images: List, file_id: str,
                        document_prefix: Optional[str] = None) -> List:
        """Upload images to object storage and return stored images."""
        stored_images = []
        for img in images:
            try:
                storage_uri = self.image_repository.put_image(
                    document_id=file_id,
                    image_id=img.image_id,
                    image_bytes=img.image_bytes,
                    image_format=img.format,
                    document_prefix=document_prefix,
                )
                img.storage_path = storage_uri
                stored_images.append(img)
            except Exception as e:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_STORAGE,
                    f"Failed to upload image {img.image_id}: {e}",
                )
        return stored_images

    def _get_page_count(self, file_path: str) -> int:
        """Get total page count of a PDF document."""
        from rag_lib.utils.pdf_utils import get_pdf_page_count
        return get_pdf_page_count(file_path)

    def generate_descriptions(
        self,
        images: List,
        model_id: str = None,
        max_workers: int = 5,
        classify_infographics: bool = True,
    ) -> List:
        """
        Generate AI descriptions for extracted images using the Vision LLM.

        When *classify_infographics* is ``True`` (the default), the Vision LLM
        is asked to return a JSON payload that includes both the description
        **and** an ``is_infographic`` boolean flag.  This piggybacks infographic
        classification on the existing description call -- zero extra API cost.

        Uses ThreadPoolExecutor for parallel API calls (I/O-bound).

        Args:
            images: List of ExtractedImage objects with image_bytes
            model_id: LLM model ID for vision (default from config)
            max_workers: Number of parallel description workers (default: 5)
            classify_infographics: Also classify each image as infographic or not.

        Returns:
            Same list with ``description`` (and optionally ``is_infographic``)
            fields populated.
        """
        if not images:
            return images


        if classify_infographics:
            prompt = (
                "You are a document image analysis expert. Analyse this image and "
                "respond with ONLY a valid JSON object (no markdown fences):\n"
                '{"description": "<detailed structured description — see rules below>", '
                '"is_infographic": true or false, '
                '"is_decorative": true or false}\n\n'
                "DESCRIPTION RULES — extract ALL visible content into structured text:\n"
                "1. **Title / heading**: If the image has a title or heading, state it first.\n"
                "2. **Sections / regions**: For each distinct box, area, or labelled group, "
                "use a section heading followed by its content.\n"
                "3. **Bullet points**: Reproduce every bullet exactly. Pair labels with values: "
                '"**Label:** value".\n'
                "4. **Tables**: Convert tabular content to Markdown table syntax.\n"
                "5. **Charts / graphs**: State the chart type (bar, pie, line), axis labels, "
                "legend entries, and ALL visible data values/percentages.\n"
                "6. **Diagrams / flowcharts**: Describe the structure — list each node/segment "
                "and their relationships (arrows, connections).\n"
                "7. **Numbers & metrics**: Every number, percentage, currency value, and unit "
                "MUST appear paired with its label.\n"
                "8. **Footnotes & fine print**: Capture all disclaimers, notes, and footer text.\n"
                "9. **Photos / logos**: Describe subject, visible text, and purpose.\n"
                "10. Do NOT invent data — only report what is visible.\n"
                "Be thorough — this description is used for semantic search, so include ALL "
                "searchable content. Maximum 600 words.\n\n"
                "INFOGRAPHIC = charts, graphs, diagrams, flowcharts, data visualisations, "
                "process flows, timelines, maps with data overlays, annotated screenshots, "
                "dashboards, visual summaries, tables with graphics.\n"
                "NOT infographic = logos, icons, decorative images, photos, headshots, "
                "navigation elements, banners, stock photos, simple text screenshots.\n\n"
                "DECORATIVE = logos, icons, navigation bars/headers, page borders, "
                "background patterns, separator lines, watermarks, stock photos with no "
                "informational value, generic banners, social media icons, "
                "simple repeated brand elements.\n"
                "NOT decorative = infographics, meaningful photos (people, events, products), "
                "charts, diagrams, screenshots with data, illustrations that explain concepts, "
                "maps, architectural drawings, any image with substantive information."
            )
        else:
            prompt = (
                "You are a document image analysis expert. Extract ALL visible content "
                "from this image into structured Markdown.\n\n"
                "RULES:\n"
                "1. **Title / heading**: State any title or heading first.\n"
                "2. **Sections**: For each distinct box, area, or labelled group, "
                "use a heading followed by its content.\n"
                "3. **Bullet points**: Reproduce every bullet exactly. Pair labels with values: "
                "`**Label:** value`.\n"
                "4. **Tables**: Convert tabular content to Markdown table syntax.\n"
                "5. **Charts / graphs**: State the chart type, axis labels, legend, "
                "and ALL visible data values/percentages.\n"
                "6. **Diagrams / flowcharts**: List each node and their relationships.\n"
                "7. **Numbers & metrics**: Every number, percentage, and unit "
                "MUST appear paired with its label.\n"
                "8. **Footnotes & fine print**: Capture all disclaimers and notes.\n"
                "9. **Photos / logos**: Describe subject, visible text, and purpose.\n"
                "10. Do NOT invent data — only report what is visible.\n\n"
                "Be thorough — this description is used for semantic search. "
                "Maximum 600 words. Output raw Markdown only — no preamble."
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _describe_single(img):
            if not img.image_bytes:
                return img, None, False, False
            try:
                raw = self._call_vlm(img, model_id, prompt)
                if not raw:
                    return img, None, False, False

                raw = raw.strip()

                # Try to parse JSON response (combined prompt)
                if classify_infographics:
                    desc, is_info, is_deco = self._parse_description_response(raw)
                    return img, desc, is_info, is_deco
                else:
                    return img, raw, False, False
            except Exception as e:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_STORAGE,
                    f"Failed to describe image {img.image_id}: {e}",
                )
                return img, None, False, False

        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Generating descriptions for {len(images)} images with {max_workers} parallel workers"
            f"{' (with infographic classification)' if classify_infographics else ''}",
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_describe_single, img): img for img in images}
            for future in as_completed(futures):
                img, description, is_infographic, is_decorative = future.result()
                if description:
                    img.description = description
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_STORAGE,
                        f"Image {img.image_id}: {description[:60]}..."
                        f"{' [INFOGRAPHIC]' if is_infographic else ''}"
                        f"{' [DECORATIVE]' if is_decorative else ''}",
                    )
                # Always set the flags (default False on failure / non-classification)
                if classify_infographics:
                    img.is_infographic = is_infographic
                    img.is_decorative = is_decorative

        return images

    @staticmethod
    def _parse_description_response(raw: str) -> tuple:
        """Parse the combined description + infographic + decorative JSON response.

        Returns:
            (description, is_infographic, is_decorative) tuple.  Falls back gracefully
            if the LLM doesn't return valid JSON.
        """
        from rag_lib.utils.json_utils import parse_llm_json

        data = parse_llm_json(raw)
        if data is not None:
            desc = data.get("description", "").strip()
            is_info = bool(data.get("is_infographic", False))
            is_deco = bool(data.get("is_decorative", False))
            return desc if desc else raw, is_info, is_deco

        # Fallback: treat entire response as description, no classification
        LOGGER.log_warning(
            ProcessLogger.STAGE_STORAGE,
            f"Could not parse JSON from vision response, using raw text: {raw[:80]}",
        )
        return raw, False, False

    def _call_vlm(self, img, model_id: str, prompt: str) -> Optional[str]:
        """Call the LLM Converse API for image description."""
        valid_formats = ["png", "jpeg", "gif", "webp"]
        img_format = img.format.lower() if img.format.lower() in valid_formats else "png"

        return self._converse.converse(
            model_id=model_id,
            prompt=prompt,
            image_bytes=img.image_bytes,
            image_format=img_format,
            max_tokens=1500,
            temperature=0,
        )
