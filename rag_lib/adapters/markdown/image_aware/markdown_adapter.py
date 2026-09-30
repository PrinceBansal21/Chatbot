"""
Image-Aware Markdown Adapter

A wrapper adapter that adds image extraction and storage capabilities
to any existing markdown adapter (Docling, Hybrid, VisionLLM, etc.).

Features:
- Extracts images from PDF documents
- Stores images to the local filesystem or configured cloud object storage
- Optionally generates image descriptions using Vision LLM
- Embeds image references in the markdown output
- Creates image manifest for vector DB metadata
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_lib.interfaces.markdown_manager import MarkdownManager
from .models import (
    ExtractedImage,
    ImageManifest,
    MarkdownWithImages,
    BoundingBox,
    ImageType,
)
from .image_extractor import ImageExtractor

# Try to import HTML image extractor
try:
    from .html_image_extractor import HtmlImageExtractor
    HTML_EXTRACTOR_AVAILABLE = True
except ImportError:
    HtmlImageExtractor = None
    HTML_EXTRACTOR_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class ImageAwareMarkdownAdapter(MarkdownManager):
    """
    Wrapper adapter that adds image extraction capabilities to any markdown adapter.
    
    This adapter:
    1. Uses the base adapter for markdown conversion
    2. Extracts images from the document
    3. Optionally generates descriptions for images using Vision LLM
    4. Stores images to configured storage location
    5. Embeds image references in the markdown
    6. Returns MarkdownWithImages containing both markdown and image metadata
    
    Example usage:
        from rag_lib.adapters.markdown.hybrid import HybridAdapter
        from rag_lib.adapters.markdown.image_aware import ImageAwareMarkdownAdapter
        
        # Create base adapter
        base_adapter = HybridAdapter(vision_model="nova-pro")
        
        # Wrap with image-aware capabilities
        adapter = ImageAwareMarkdownAdapter(
            base_adapter=base_adapter,
            storage_path="/path/to/images",
            enable_descriptions=True,
        )
        
        # Convert document
        result = adapter.convert_with_images("document.pdf")
        print(result.markdown)
        print(result.images.to_dict())
    """
    
    def __init__(
        self,
        base_adapter: Optional[MarkdownManager] = None,
        base_adapter_type: str = "hybrid",
        base_adapter_config: Optional[Dict[str, Any]] = None,
        storage_path: Optional[str] = None,
        storage_type: str = "local",  # "local" or "cloud"
        storage_bucket: Optional[str] = None,
        path_prefix: Optional[str] = None,
        enable_descriptions: bool = True,
        description_model: Optional[str] = None,
        region: Optional[str] = None,
        extract_full_pages: bool = False,
        min_image_width: int = 50,
        min_image_height: int = 50,
        max_image_area_ratio: float = 0.85,
        embed_images_in_markdown: bool = True,
        image_reference_format: str = "markdown",  # "markdown", "html", "placeholder"
        save_manifest: bool = True,
        # HTML-specific options
        fetch_remote_images: bool = False,
        html_asset_root: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the image-aware adapter.
        
        Args:
            base_adapter: Existing markdown adapter to wrap (optional)
            base_adapter_type: Type of adapter to create if base_adapter not provided
                Options: "docling", "hybrid", "vision_llm"
            base_adapter_config: Config for creating base adapter
            storage_path: Local path for image storage
            storage_type: "local" or "cloud"
            storage_bucket: Object-storage container name (if storage_type="cloud")
            path_prefix: Object key prefix (if storage_type="cloud")
            enable_descriptions: Generate image descriptions using Vision LLM
            description_model: Model for image descriptions
            region: Cloud region (defaults to provider configuration when omitted)
            extract_full_pages: Also extract full page renders
            min_image_width: Minimum image width to extract
            min_image_height: Minimum image height to extract
            max_image_area_ratio: Maximum ratio of image area to page area.
                Images covering more of the page than this (e.g. full-page
                scanned backgrounds) are skipped as redundant with the
                already-extracted text content. Set to 1.0 to disable.
            embed_images_in_markdown: Add image references to markdown
            image_reference_format: Format for image references
            save_manifest: Save image manifest JSON file
        """
        self.config = kwargs
        
        # Storage configuration
        self.storage_path = storage_path
        self.storage_type = storage_type
        self.storage_bucket = storage_bucket
        self.path_prefix = path_prefix or "images"
        
        # Description configuration
        self.enable_descriptions = enable_descriptions
        self.description_model = description_model
        self.region = region
        
        # Extraction configuration
        self.extract_full_pages = extract_full_pages
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_image_area_ratio = max_image_area_ratio
        
        # Output configuration
        self.embed_images_in_markdown = embed_images_in_markdown
        self.image_reference_format = image_reference_format
        self.save_manifest = save_manifest
        
        # HTML-specific configuration
        self.fetch_remote_images = fetch_remote_images
        self.html_asset_root = html_asset_root
        
        # Initialize base adapter
        self._base_adapter = base_adapter
        self._base_adapter_type = base_adapter_type
        self._base_adapter_config = base_adapter_config or {}
        
        # Initialize PDF image extractor
        self._pdf_extractor = ImageExtractor(
            min_width=min_image_width,
            min_height=min_image_height,
            max_area_ratio=max_image_area_ratio,
            extract_full_pages=extract_full_pages,
        )
        
        # Initialize HTML image extractor (if available)
        self._html_extractor = None
        if HTML_EXTRACTOR_AVAILABLE:
            try:
                self._html_extractor = HtmlImageExtractor(
                    min_width=min_image_width,
                    min_height=min_image_height,
                    fetch_remote=fetch_remote_images,
                )
            except ImportError as e:
                LOGGER.warning("HTML image extractor not available: %s", e)
        
        # Lazy-loaded components
        self._file_manager = None
        self._vision_llm = None
        self._llm_adapter = None
        
        LOGGER.info(
            "ImageAwareMarkdownAdapter initialized: storage_type=%s, "
            "enable_descriptions=%s, embed_images=%s",
            storage_type, enable_descriptions, embed_images_in_markdown
        )
    
    @property
    def base_adapter(self) -> MarkdownManager:
        """Lazy-load the base adapter."""
        if self._base_adapter is None:
            self._base_adapter = self._create_base_adapter()
        return self._base_adapter
    
    def _create_base_adapter(self) -> MarkdownManager:
        """Create the base adapter based on configuration."""
        from rag_lib.factory import RAGFactory
        
        adapter_type = self._base_adapter_type
        config = self._base_adapter_config.copy()
        
        # Apply common config
        if self.region:
            config.setdefault("region", self.region)
        
        LOGGER.info("Creating base adapter: %s", adapter_type)
        return RAGFactory.get_markdown_manager(adapter_type, **config)
    
    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """
        Convert document to markdown (standard interface).
        
        This method returns only the markdown string for compatibility
        with the MarkdownManager interface. Use convert_with_images()
        to get the full result with image metadata.
        """
        result = self.convert_with_images(file_path, mime_type, **kwargs)
        return result.markdown
    
    def convert_with_images(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
        document_id: Optional[str] = None,
        asset_root: Optional[str] = None,
        **kwargs,
    ) -> MarkdownWithImages:
        """
        Convert document to markdown with image extraction.
        
        Args:
            file_path: Path to the document
            mime_type: Optional MIME type
            document_id: Optional document ID (auto-generated if not provided)
            asset_root: Root directory for resolving relative image paths (HTML only)
            **kwargs: Additional arguments for base adapter
            
        Returns:
            MarkdownWithImages containing markdown and image metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate document ID
        if document_id is None:
            document_id = self._generate_document_id(file_path)
        
        LOGGER.info("Converting document with images: %s (id=%s)", path.name, document_id)
        
        # Step 1: Convert to markdown using base adapter
        markdown = self.base_adapter.convert(file_path, mime_type, **kwargs)
        
        # Step 2: Extract images from document (pass asset_root for HTML)
        images = self._extract_images(file_path, document_id, asset_root=asset_root)
        
        # Step 3: Optionally generate descriptions
        if self.enable_descriptions and images:
            images = self._describe_images(images, file_path)
        
        # Step 4: Store images
        if images:
            images = self._store_images(images, document_id)
        
        # Step 5: Create image manifest
        manifest = ImageManifest(
            document_id=document_id,
            source_file=str(path.absolute()),
            images=images,
            total_pages=self._get_page_count(file_path),
            extraction_timestamp=datetime.utcnow().isoformat(),
        )
        
        # Step 6: Optionally embed image references in markdown
        if self.embed_images_in_markdown and images:
            markdown = self._embed_image_references(markdown, manifest)
        
        # Step 7: Save manifest
        if self.save_manifest and self.storage_path:
            self._save_manifest(manifest, document_id)
        
        result = MarkdownWithImages(
            markdown=markdown,
            images=manifest,
            document_id=document_id,
            source_file=str(path.absolute()),
        )
        
        LOGGER.info(
            "Conversion complete: %d images extracted, markdown length=%d",
            len(images), len(markdown)
        )
        
        return result
    
    def _extract_images(
        self,
        file_path: str,
        document_id: str,
        asset_root: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from the document.
        
        Supports both PDF and HTML files:
        - PDF: Uses PyMuPDF to extract embedded images
        - HTML: Parses <img> tags and loads images from paths/URLs
        
        Args:
            file_path: Path to the document
            document_id: Unique document identifier
            asset_root: Root directory for resolving relative image paths (HTML only)
        """
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.pdf':
                # Use PDF extractor
                images = self._pdf_extractor.extract_images(
                    file_path=file_path,
                    document_id=document_id,
                )
            elif ext in ('.html', '.htm'):
                # Use HTML extractor
                if self._html_extractor is None:
                    LOGGER.warning("HTML image extractor not available. Install beautifulsoup4.")
                    return []
                
                # Determine asset root for resolving relative image paths
                effective_asset_root = asset_root or self.html_asset_root or str(Path(file_path).parent)
                
                images = self._html_extractor.extract_images(
                    html_path=file_path,
                    asset_root=effective_asset_root,
                    document_id=document_id,
                )
            else:
                LOGGER.debug("No image extraction support for file type: %s", ext)
                return []
            
            LOGGER.info("Extracted %d images from document", len(images))
            return images
        except Exception as e:
            LOGGER.warning("Failed to extract images: %s", e)
            return []
    
    def _describe_images(
        self,
        images: List[ExtractedImage],
        file_path: str,
    ) -> List[ExtractedImage]:
        """Generate descriptions for images using the configured Vision LLM."""
        if not images:
            return images
        
        LOGGER.info("Generating descriptions for %d images using %s", len(images), self.description_model)
        
        described_images = []
        for img in images:
            try:
                LOGGER.debug("Describing image: %s", img.image_id)
                description = self._generate_image_description(img)
                img.description = description
                LOGGER.info("Image %s described: %s...", img.image_id, description[:50] if description else "None")
            except Exception as e:
                LOGGER.warning(
                    "Failed to describe image %s: %s",
                    img.image_id, e
                )
            described_images.append(img)
        
        return described_images
    
    def _generate_image_description(self, image: ExtractedImage) -> str:
        """Generate a description for a single image using the cloud_lib LLMManager."""
        from cloud_lib.factory import CloudFactory

        if self._llm_adapter is None:
            self._llm_adapter = CloudFactory.get_llm_manager(model_key="vision")
        img_format = image.format.lower()
        if img_format == "jpg":
            img_format = "jpeg"
        
        prompt = """You are a document image analysis expert. Extract ALL visible content \
from this image into structured Markdown for RAG (Retrieval Augmented Generation).

RULES:
1. **Title / heading**: If the image has a title or heading, state it first.
2. **Sections / regions**: For each distinct box, area, or labelled group, \
use a `##` heading followed by its content.
3. **Bullet points**: Reproduce every bullet exactly. Pair labels with values: \
`- **Label:** value`.
4. **Tables**: Convert tabular content to proper Markdown tables with headers.
5. **Charts / graphs**: State the chart type (bar, pie, line, etc.), axis labels, \
legend entries, and ALL visible data values, percentages, and units.
6. **Diagrams / flowcharts**: Describe the structure — list each node/segment \
and their relationships (arrows, connections, parent-child).
7. **Numbers & metrics**: Every number, percentage, currency value, and unit \
MUST appear paired with its label on the same line.
8. **Footnotes & fine print**: Capture all disclaimers, notes, and footer text.
9. **Photos / logos**: Describe the subject, any visible text, and its purpose \
in the document context.
10. Do NOT invent data — only report what is visible.

Be thorough — this description will be used for semantic search, so include ALL \
searchable content. Maximum 600 words. Prioritize data extraction over general \
descriptions. Output raw Markdown only — no preamble or commentary."""

        description = self._llm_adapter.converse_vision(
            model_id=self.description_model,
            prompt=prompt,
            image_bytes=image.image_bytes,
            image_format=img_format,
            max_tokens=1500,
            temperature=0,
        )
        return (description or "").strip()
    
    def _store_images(
        self,
        images: List[ExtractedImage],
        document_id: str,
    ) -> List[ExtractedImage]:
        """Store images to configured storage location."""
        if not images:
            return images
        
        if self.storage_type == "cloud":
            return self._store_images_cloud(images, document_id)
        else:
            return self._store_images_local(images, document_id)
    
    def _store_images_local(
        self,
        images: List[ExtractedImage],
        document_id: str,
    ) -> List[ExtractedImage]:
        """Store images to local filesystem."""
        if not self.storage_path:
            LOGGER.warning("No storage path configured, skipping image storage")
            return images
        
        # Create storage directory
        storage_dir = Path(self.storage_path) / document_id
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        stored_images = []
        for img in images:
            try:
                # Generate filename
                filename = f"{img.image_id}.{img.format}"
                file_path = storage_dir / filename
                
                # Write image
                with open(file_path, "wb") as f:
                    f.write(img.image_bytes)
                
                # Update image metadata
                img.storage_path = str(file_path.absolute())
                img.storage_url = f"file://{file_path.absolute()}"
                
                LOGGER.debug("Stored image: %s", file_path)
                
            except Exception as e:
                LOGGER.warning("Failed to store image %s: %s", img.image_id, e)
            
            stored_images.append(img)
        
        LOGGER.info("Stored %d images to %s", len(stored_images), storage_dir)
        return stored_images
    
    def _store_images_cloud(
        self,
        images: List[ExtractedImage],
        document_id: str,
    ) -> List[ExtractedImage]:
        """Store images with the configured cloud object-storage adapter."""
        if not self.storage_bucket:
            LOGGER.warning("No object-storage container configured, skipping image storage")
            return images
        
        try:
            from cloud_lib.factory import CloudFactory
            
            if self._file_manager is None:
                self._file_manager = CloudFactory.get_file_manager()
            
            stored_images = []
            for img in images:
                try:
                    file_key = f"{self.path_prefix}/{document_id}/{img.image_id}.{img.format}"
                    
                    self._file_manager.put_object(
                        self.storage_bucket,
                        file_key,
                        img.image_bytes,
                        content_type=f"image/{img.format}",
                    )
                    
                    # Update image metadata
                    img.storage_path = self._file_manager.build_uri(
                        self.storage_bucket, file_key
                    )
                    img.storage_url = self._file_manager.get_object_url(
                        self.storage_bucket, file_key
                    )
                    
                    LOGGER.debug("Uploaded image to cloud object storage: %s", file_key)
                    
                except Exception as e:
                    LOGGER.warning(
                        "Failed to upload image %s to cloud object storage: %s",
                        img.image_id,
                        e,
                    )
                
                stored_images.append(img)
            
            LOGGER.info("Uploaded %d images to cloud object storage", len(stored_images))
            return stored_images
            
        except ImportError:
            LOGGER.warning("cloud_lib not available, cannot upload images to object storage")
            return images
    
    def _embed_image_references(
        self,
        markdown: str,
        manifest: ImageManifest,
    ) -> str:
        """Embed image references in the markdown."""
        if not manifest.images:
            return markdown
        
        # Group images by page
        images_by_page = {}
        for img in manifest.images:
            if img.page_number not in images_by_page:
                images_by_page[img.page_number] = []
            images_by_page[img.page_number].append(img)
        
        # Find page markers and insert image references
        lines = markdown.split("\n")
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            
            # Check for page marker
            if line.strip().startswith("<!-- page_number:"):
                try:
                    # Extract page number
                    page_str = line.split(":")[1].strip().rstrip("-->").strip()
                    page_num = int(page_str)
                    
                    # Add image references for this page
                    if page_num in images_by_page:
                        new_lines.append("")
                        new_lines.append("<!-- Images on this page -->")
                        for img in images_by_page[page_num]:
                            ref = self._format_image_reference(img)
                            new_lines.append(ref)
                        new_lines.append("")
                except (ValueError, IndexError):
                    pass
        
        return "\n".join(new_lines)
    
    def _format_image_reference(self, image: ExtractedImage) -> str:
        """Format an image reference based on configured format."""
        alt_text = image.description or f"Image {image.image_id}"
        url = image.storage_url or image.storage_path or f"[{image.image_id}]"
        
        if self.image_reference_format == "markdown":
            return f"![{alt_text}]({url})"
        elif self.image_reference_format == "html":
            return f'<img src="{url}" alt="{alt_text}" />'
        else:  # placeholder
            desc = f" - {image.description[:100]}..." if image.description else ""
            return f"<!-- image: {image.image_id} (page {image.page_number}){desc} -->"
    
    def _save_manifest(self, manifest: ImageManifest, document_id: str) -> None:
        """Save image manifest to JSON file."""
        if not self.storage_path:
            return
        
        try:
            manifest_dir = Path(self.storage_path) / document_id
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "manifest.json"
            
            with open(manifest_path, "w") as f:
                json.dump(manifest.to_dict(), f, indent=2)
            
            LOGGER.info("Saved image manifest: %s", manifest_path)
            
        except Exception as e:
            LOGGER.warning("Failed to save manifest: %s", e)
    
    def _get_page_count(self, file_path: str) -> int:
        """Get total page count of document."""
        from rag_lib.utils.pdf_utils import get_pdf_page_count
        return get_pdf_page_count(file_path)
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID."""
        path = Path(file_path)
        name = path.stem
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{name}_{unique_suffix}"
