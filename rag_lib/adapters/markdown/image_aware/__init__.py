"""
Image-Aware Markdown Adapter Module

This module provides image extraction and storage capabilities
for markdown conversion pipelines.

Components:
- ImageAwareMarkdownAdapter: Wrapper adapter for any markdown adapter
- ImageExtractor: Extracts images from PDF documents
- HtmlImageExtractor: Extracts images from HTML documents
- Models: Data classes for images and manifests

Usage:
    from rag_lib.adapters.markdown.image_aware import (
        ImageAwareMarkdownAdapter,
        ImageExtractor,
        HtmlImageExtractor,
        ExtractedImage,
        ImageManifest,
        MarkdownWithImages,
    )
    
    # Wrap an existing adapter
    adapter = ImageAwareMarkdownAdapter(
        base_adapter_type="hybrid",
        storage_path="/path/to/images",
        enable_descriptions=True,
    )
    
    # Convert PDF with image extraction
    result = adapter.convert_with_images("document.pdf")
    
    # Convert HTML with image extraction
    result = adapter.convert_with_images("page.html", asset_root="./assets/")
"""

from .models import (
    ExtractedImage,
    ImageManifest,
    MarkdownWithImages,
    BoundingBox,
    ImageType,
)
from .image_extractor import ImageExtractor
from .base_extractor import BaseImageExtractor
from .markdown_adapter import ImageAwareMarkdownAdapter

# Optional HTML extractor (requires beautifulsoup4)
try:
    from .html_image_extractor import HtmlImageExtractor
    _HTML_AVAILABLE = True
except ImportError:
    HtmlImageExtractor = None
    _HTML_AVAILABLE = False

# Office document extractors (DOCX / PPTX — no extra dependencies)
from .office_image_extractor import DocxImageExtractor, PptxImageExtractor

__all__ = [
    # Main adapter
    "ImageAwareMarkdownAdapter",
    
    # Image extraction
    "BaseImageExtractor",
    "ImageExtractor",
    "HtmlImageExtractor",
    "DocxImageExtractor",
    "PptxImageExtractor",
    
    # Data models
    "ExtractedImage",
    "ImageManifest",
    "MarkdownWithImages",
    "BoundingBox",
    "ImageType",
]
