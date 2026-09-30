"""Document processing stage: fetch, convert, vision, PII, summary, images."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from .document_fetcher import DocumentFetcher, FetchedDocumentBundle
from .markdown import MarkdownConverter, MarkdownRepository
from services.image import ImageProcessor, MarkdownImageEmbedder
from services.llm import SummaryGenerator, ImageDescriptionAugmenter

from utils import (
    get_source_uri,
    ImageConfigResolver,
    MetadataEnricher,
)
from utils.ole2_converter import OLE2Converter
from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class DocumentProcessor:
    """
    Handles the document-processing pipeline stages:
      fetch → convert → vision → PII → summary → image extraction → store markdown.

    Accepts pre-built service instances so it can be created inside Pipeline.__init__
    without changing how services are constructed.
    """

    def __init__(
        self,
        fetcher: DocumentFetcher,
        converter: MarkdownConverter,
        repository: MarkdownRepository,
        *,
        vision: Optional[ImageDescriptionAugmenter] = None,
        pii_manager=None,
        summary_generator: Optional[SummaryGenerator] = None,
        image_processor: Optional[ImageProcessor] = None,
        image_config_resolver: Optional[ImageConfigResolver] = None,
        markdown_preprocessor=None,
        image_extraction_enabled: bool = False,
        summary_enabled: bool = False,
        page_summaries_enabled: bool = False,
    ) -> None:
        self.fetcher = fetcher
        self.converter = converter
        self.repository = repository
        self.vision = vision
        self.pii_manager = pii_manager
        self.summary_generator = summary_generator
        self.image_processor = image_processor
        self.image_config_resolver = image_config_resolver
        self.markdown_preprocessor = markdown_preprocessor
        self.image_extraction_enabled = image_extraction_enabled
        self.summary_enabled = summary_enabled
        self.page_summaries_enabled = page_summaries_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        payload: Dict,
        file_id: str,
        metadata: Dict[str, str],
        strategies: Dict[str, str],
        file_type_config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Process document: fetch, convert, and optionally extract images.

        Returns:
            Tuple of (markdown, markdown_location, image_manifest, fetched_bundle).
            The caller owns *fetched_bundle* and must call
            ``self.fetcher.cleanup(fetched_bundle)`` when done.
        """
        provided_markdown_location = payload.get("markdown_location")
        image_manifest = None
        fetched_bundle = None
        file_type_config = file_type_config or {}

        if provided_markdown_location:
            markdown, markdown_location = self._use_existing_markdown(provided_markdown_location)
        else:
            markdown, markdown_location, image_manifest, fetched_bundle = (
                self._process_new_document(
                    payload, file_id, metadata, strategies, file_type_config
                )
            )

        return markdown, markdown_location, image_manifest, fetched_bundle

    def prepare_for_chunking(
        self,
        markdown: str,
        payload: Dict,
        file_type_config: Dict[str, Any],
    ) -> str:
        """Delegate to the markdown preprocessor."""
        return self.markdown_preprocessor.prepare_for_chunking(
            markdown, payload, file_type_config
        )

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _use_existing_markdown(self, markdown_location: str) -> tuple:
        """Use pre-existing markdown file."""
        LOGGER.log_stage(ProcessLogger.STAGE_FETCH, f"Using existing markdown from: {markdown_location}")
        markdown = self._fetch_existing_markdown(markdown_location)
        LOGGER.log_stage(ProcessLogger.STAGE_CONVERT, f"Loaded {len(markdown)} chars from existing markdown")
        return markdown, markdown_location

    def _process_new_document(
        self,
        payload: Dict,
        file_id: str,
        metadata: Dict[str, str],
        strategies: Dict[str, str],
        file_type_config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Process a new document: fetch, convert, and store."""
        source_uri = get_source_uri(payload)
        if not source_uri:
            raise ValueError(
                "Payload missing 'source_uri', 'storage_location', 'web_url', or 'markdown_location'"
            )

        # Derive document_prefix for co-located storage.
        # If the source is an object-storage URI like
        #   scheme://container/knowledgebases/kb/sgt/sgi/doc/file_id/report.pdf
        # we strip the bucket and the filename to get:
        #   knowledgebases/kb/sgt/sgi/doc/file_id
        # Markdown and images will be stored alongside the document.
        document_prefix = self._derive_document_prefix(source_uri)

        file_type_config = file_type_config or {}

        # Fetch document
        LOGGER.log_fetch(source_uri)
        fetched_bundle = self.fetcher.download(source_uri, payload.get("file_type"))
        LOGGER.log_fetch_complete(fetched_bundle.document_path)

        # Auto-convert OLE2 .doc to DOCX for enhanced image extraction
        # Docling can extract images from DOCX/PDF but not from legacy OLE2 format
        converted_path = OLE2Converter.convert_ole2_to_docx(fetched_bundle.document_path)
        if converted_path != fetched_bundle.document_path:
            # Update the bundle to use the converted file
            fetched_bundle.document_path = converted_path
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"Using converted DOCX for image extraction: {converted_path}"
            )

        # Convert to markdown with file-type-specific configuration
        markdown_config = file_type_config.get("markdown", {})
        image_config = file_type_config.get("image_extraction", {})

        # For PPTX files: persist rendered slide images as they are produced
        # to avoid retaining large image bytes for all slides in memory.
        slide_manifest_images: list = []
        convert_kwargs: Dict[str, Any] = {}
        is_pptx = markdown_config.get("provider") == "pptx_vision"
        store_slides = image_config.get("store_slide_images", True)

        if is_pptx and self.image_processor and store_slides:
            def _collect_slide(page_num: int, png_bytes: bytes, slide_md: str, slide_type: str = "content"):
                image_id = f"slide-{page_num:03d}"
                try:
                    storage_uri = self.image_processor.image_repository.put_image(
                        document_id=file_id,
                        image_id=image_id,
                        image_bytes=png_bytes,
                        image_format="png",
                        document_prefix=document_prefix,
                    )
                    slide_manifest_images.append({
                        "image_id": image_id,
                        "storage_path": storage_uri,
                        "page": str(page_num),
                        "description": (slide_md or "")[:500],
                        "format": "png",
                        "image_type": "slide",
                        "slide_type": slide_type,
                    })
                except Exception as e:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_STORAGE,
                        f"Failed to upload slide image {image_id}: {e}",
                    )
            convert_kwargs["slide_image_callback"] = _collect_slide

        markdown = self._convert_to_markdown(
            fetched_bundle, payload, strategies["markdown_provider"],
            markdown_config, **convert_kwargs,
        )

        # Vision/image description
        markdown = self._apply_vision(
            markdown, fetched_bundle, file_id, strategies["vision_provider"]
        )

        # PII redaction
        markdown = self._apply_pii_redaction(markdown)

        # --- Parallel: summary generation + image extraction ---
        # These are independent: summaries read the markdown, image extraction
        # reads the PDF.  Run them concurrently to save IO wait time.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="doc_proc") as pool:
            summary_future = pool.submit(
                self._generate_document_summary, markdown, metadata.get("file_name")
            )
            image_future = pool.submit(
                self._extract_images, fetched_bundle, file_id, metadata,
                payload, image_config, document_prefix,
            )

            # Collect results
            summary = summary_future.result()
            if summary:
                self.summary_generator.add_summary_to_metadata(metadata, summary)

            image_manifest = image_future.result()

        # Merge uploaded slide images (PPTX) into the image manifest.
        if slide_manifest_images:
            slide_manifest = {
                "document_id": file_id,
                "source_file": f"{file_id} (slide renders)",
                "total_pages": len(slide_manifest_images),
                "image_count": len(slide_manifest_images),
                "images": slide_manifest_images,
            }
            if image_manifest:
                existing_images = image_manifest.get("images", [])
                image_manifest["images"] = existing_images + slide_manifest_images
                LOGGER.log_stage(
                    ProcessLogger.STAGE_STORAGE,
                    f"Merged {len(slide_manifest_images)} slide images into manifest "
                    f"({len(existing_images)} extracted images + {len(slide_manifest_images)} slides)",
                )
            else:
                image_manifest = slide_manifest

        # Generate per-page summaries (stored in metadata as JSON for chunk enrichment)
        if self.page_summaries_enabled and self.summary_generator:
            page_summaries = self.summary_generator.generate_page_summaries(markdown)
            if page_summaries:
                metadata["page_summaries"] = json.dumps(page_summaries)
                LOGGER.log_stage(
                    ProcessLogger.STAGE_CONVERT,
                    f"Generated page-level summaries for {len(page_summaries)} pages",
                )

        # Update metadata with image info
        if image_manifest:
            self._update_metadata_with_images(metadata, image_manifest)
            markdown = self._embed_images_in_markdown(markdown, image_manifest)
            MarkdownImageEmbedder.enrich_manifest_with_context(markdown, image_manifest)

        # Store markdown
        LOGGER.log_storage(self.repository.bucket, f"{file_id}/content.md")
        markdown_location = self.repository.put(
            file_id=file_id, markdown=markdown, metadata=metadata,
            document_prefix=document_prefix,
        )

        return markdown, markdown_location, image_manifest, fetched_bundle

    def _convert_to_markdown(
        self,
        fetched_bundle: FetchedDocumentBundle,
        payload: Dict,
        markdown_provider: str,
        markdown_config: Optional[Dict[str, Any]] = None,
        **convert_kwargs,
    ) -> str:
        """Convert document to markdown."""
        LOGGER.log_conversion(markdown_provider, fetched_bundle.document_path)

        # Extract provider-specific config from the file-type markdown config
        provider_config = None
        if markdown_config:
            provider_config = markdown_config.get(markdown_provider, {})
            for key, value in markdown_config.items():
                if key not in ["provider", "default_provider"] and not isinstance(value, dict):
                    if provider_config is None:
                        provider_config = {}
                    provider_config[key] = value

        markdown = self.converter.convert(
            fetched_bundle.document_path,
            payload.get("mime_type"),
            provider=markdown_provider,
            provider_config=provider_config,
            **convert_kwargs,
        )

        # Capture auto-detected infographic pages for downstream use
        self.detected_infographic_pages = getattr(
            self.converter, "detected_infographic_pages", []
        )

        LOGGER.log_conversion_complete(len(markdown))
        return markdown

    def _apply_vision(
        self,
        markdown: str,
        fetched_bundle: FetchedDocumentBundle,
        file_id: str,
        vision_provider: Optional[str],
    ) -> str:
        """Apply vision/image description to markdown."""
        if not self.vision:
            return markdown

        LOGGER.log_vision(vision_provider or self.vision.default_provider, 0)
        markdown = self.vision.annotate_markdown(
            markdown,
            asset_root=fetched_bundle.asset_root,
            file_id=file_id,
            source_path=fetched_bundle.document_path,
            provider=vision_provider,
        )
        LOGGER.log_vision_complete(0)
        return markdown

    @staticmethod
    def _derive_document_prefix(source_uri: str) -> Optional[str]:
        """Derive the object-storage key prefix (directory) of the original document.

        Given a provider-native object-storage URI ending in
        ``/knowledgebases/kb/sgt/sgi/doc/fid/report.pdf``
        returns ``knowledgebases/kb/sgt/sgi/doc/fid``.

        For non-object-storage sources (web URLs, local paths) returns ``None``
        so callers use the default output layout.
        """
        if not source_uri:
            return None
        parsed = urlsplit(source_uri)
        if not parsed.scheme or parsed.scheme in ("http", "https") or not parsed.netloc:
            return None
        key = parsed.path.lstrip("/")
        # Strip the filename to get the directory
        if "/" not in key:
            return None
        prefix = key.rsplit("/", 1)[0]
        return prefix if prefix else None

    def _apply_pii_redaction(self, markdown: str) -> str:
        """Apply PII redaction if enabled."""
        if self.pii_manager:
            LOGGER.log_pii_redaction(True)
            markdown = self.pii_manager.redact(markdown)
            LOGGER.log_pii_complete()
        else:
            LOGGER.log_pii_redaction(False)
        return markdown

    def _generate_document_summary(
        self,
        markdown: str,
        file_name: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a document-level summary for improved retrieval."""
        if not self.summary_enabled or not self.summary_generator:
            return None
        return self.summary_generator.generate_summary(markdown, file_name=file_name)

    def _extract_images(
        self,
        fetched_bundle: FetchedDocumentBundle,
        file_id: str,
        metadata: Dict[str, str],
        payload: Dict,
        image_config: Optional[Dict[str, Any]] = None,
        document_prefix: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract and store images from document."""
        image_config = image_config or {}

        if not self.image_extraction_enabled and not image_config.get("enabled"):
            return None
        if "enabled" in image_config and not image_config["enabled"]:
            return None
        if not self.image_processor:
            return None

        payload_image_cfg = payload.get("image_extraction", {})
        merged_image_cfg = {**image_config, **payload_image_cfg}

        return self.image_processor.extract_and_store_images(
            file_path=fetched_bundle.document_path,
            file_id=file_id,
            metadata=metadata,
            asset_root=fetched_bundle.asset_root,
            payload_image_cfg=merged_image_cfg,
            document_prefix=document_prefix,
        )

    def _update_metadata_with_images(
        self,
        metadata: Dict[str, str],
        image_manifest: Dict[str, Any],
    ) -> None:
        """Update metadata with image extraction information."""
        MetadataEnricher.add_image_metadata(metadata, image_manifest)

    def _embed_images_in_markdown(
        self,
        markdown: str,
        image_manifest: Dict[str, Any],
    ) -> str:
        """Embed image URLs into markdown."""
        if self.image_config_resolver.should_embed_in_markdown():
            return MarkdownImageEmbedder.embed_image_urls(markdown, image_manifest)
        return markdown

    def _fetch_existing_markdown(self, markdown_location: str) -> str:
        """Fetch existing markdown content from a storage URI or local file path."""
        try:
            if "://" not in markdown_location:
                import os

                if not os.path.isfile(markdown_location):
                    raise FileNotFoundError(f"Local markdown file not found: {markdown_location}")
                with open(markdown_location, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            else:
                fetched = self.fetcher.download(markdown_location, file_type="markdown")
                with open(fetched.document_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                self.fetcher.cleanup(fetched)

            if not markdown_content:
                raise ValueError(f"Markdown file at {markdown_location} is empty")

            return markdown_content
        except Exception as e:
            raise ValueError(f"Failed to fetch markdown from {markdown_location}: {e}")
