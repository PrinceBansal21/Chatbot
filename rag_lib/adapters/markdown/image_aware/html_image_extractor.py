"""
HTML Image extraction utilities.
Extracts images from HTML files for image-aware RAG processing.
"""

import base64
import hashlib
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

from .models import ExtractedImage, BoundingBox, ImageType
from .base_extractor import BaseImageExtractor

LOGGER = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BEAUTIFULSOUP_AVAILABLE = False

# Optional: SVG to PNG conversion
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    cairosvg = None
    CAIROSVG_AVAILABLE = False


class HtmlImageExtractor(BaseImageExtractor):
    """
    Extract images from HTML files.
    
    Supports:
    - Relative paths: <img src="images/chart.png">
    - Absolute paths: <img src="/var/data/img.jpg">
    - Data URLs: <img src="data:image/png;base64,...">
    - Remote URLs: <img src="https://example.com/img.png"> (optional)
    
    Example:
        extractor = HtmlImageExtractor()
        images = extractor.extract_images(
            "page.html",
            asset_root="./downloaded_assets/",
            document_id="doc123"
        )
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'}
    
    # MIME type to extension mapping
    MIME_TO_EXT = {
        'image/png': 'png',
        'image/jpeg': 'jpeg',
        'image/jpg': 'jpeg',
        'image/gif': 'gif',
        'image/webp': 'webp',
        'image/bmp': 'bmp',
        'image/svg+xml': 'svg',
    }
    
    # Formats supported by Vision LLMs (for AI descriptions)
    VISION_LLM_FORMATS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        fetch_remote: bool = False,
        remote_timeout: int = 10,
        skip_decorative: bool = True,
        convert_svg_to_png: bool = True,
    ):
        """
        Initialize the HTML image extractor.
        
        Args:
            min_width: Minimum image width to extract (pixels)
            min_height: Minimum image height to extract (pixels)
            fetch_remote: Whether to fetch images from remote URLs
            remote_timeout: Timeout for remote image fetches (seconds)
            skip_decorative: Skip images marked as decorative (empty alt, role=presentation)
            convert_svg_to_png: Convert SVG images to PNG for Vision LLM compatibility
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "BeautifulSoup is required for HTML image extraction. "
                "Install with: pip install beautifulsoup4"
            )
        
        super().__init__(min_width=min_width, min_height=min_height)
        self.fetch_remote = fetch_remote
        self.remote_timeout = remote_timeout
        self.skip_decorative = skip_decorative
        self.convert_svg_to_png = convert_svg_to_png
        
        if convert_svg_to_png and not CAIROSVG_AVAILABLE:
            LOGGER.warning(
                "cairosvg not available - SVG images will not be converted to PNG. "
                "Install with: pip install cairosvg"
            )
    
    def extract_images(
        self,
        html_path: str,
        asset_root: Optional[str] = None,
        document_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from an HTML file.
        
        Args:
            html_path: Path to the HTML file
            asset_root: Root directory for resolving relative image paths.
                        If None, uses the HTML file's directory.
            document_id: Unique identifier for the document
            base_url: Base URL for resolving relative URLs (for web-scraped content)
            
        Returns:
            List of ExtractedImage objects
        """
        path = Path(html_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(html_path)
        
        # Determine asset root
        if asset_root is None:
            asset_root = str(path.parent)
        
        # Read and parse HTML
        html_content = path.read_text(encoding='utf-8', errors='ignore')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        images = []
        img_tags = soup.find_all('img')
        
        LOGGER.debug("Found %d <img> tags in %s", len(img_tags), html_path)
        
        for idx, img_tag in enumerate(img_tags):
            try:
                extracted = self._process_img_tag(
                    img_tag=img_tag,
                    index=idx,
                    asset_root=asset_root,
                    document_id=document_id,
                    base_url=base_url,
                    html_path=html_path,
                )
                if extracted:
                    images.append(extracted)
            except Exception as e:
                LOGGER.warning("Failed to extract image %d: %s", idx, e)
                continue
        
        LOGGER.info("Extracted %d images from %s", len(images), html_path)
        return images
    
    def _process_img_tag(
        self,
        img_tag: Any,
        index: int,
        asset_root: str,
        document_id: str,
        base_url: Optional[str],
        html_path: str,
    ) -> Optional[ExtractedImage]:
        """Process a single <img> tag and extract the image."""
        src = img_tag.get('src', '').strip()
        
        # If src is empty, try to extract URL from srcset attribute
        # srcset format: "url1 1x, url2 2x" or "url1 300w, url2 600w"
        if not src:
            srcset = img_tag.get('srcset', '').strip()
            if srcset:
                src = self._extract_url_from_srcset(srcset)
                if src:
                    LOGGER.debug("Extracted URL from srcset: %s", src)
        
        if not src:
            return None
        
        # Skip decorative images
        if self.skip_decorative:
            alt = img_tag.get('alt', '')
            role = img_tag.get('role', '')
            aria_hidden = img_tag.get('aria-hidden', '')
            
            # Common patterns for decorative images
            if role == 'presentation' or aria_hidden == 'true':
                LOGGER.debug("Skipping decorative image: %s", src)
                return None
        
        # Get alt text for later use
        alt_text = img_tag.get('alt', '')
        
        # Try to get dimensions from attributes
        width = self._parse_dimension(img_tag.get('width'))
        height = self._parse_dimension(img_tag.get('height'))
        
        # Determine image source type and load bytes
        image_bytes, img_format = self._load_image(
            src=src,
            asset_root=asset_root,
            base_url=base_url,
            html_path=html_path,
        )
        
        if not image_bytes:
            LOGGER.debug("Could not load image: %s", src)
            return None
        
        # Convert SVG to PNG for Vision LLM compatibility
        original_format = img_format
        if img_format == 'svg' and self.convert_svg_to_png:
            converted_bytes, converted_format = self._convert_svg_to_png(image_bytes)
            if converted_bytes:
                image_bytes = converted_bytes
                img_format = converted_format
                LOGGER.debug("Converted SVG to PNG: %s", src)
            else:
                LOGGER.warning("Failed to convert SVG to PNG, keeping original: %s", src)
        
        # Try to get actual dimensions from image data
        actual_width, actual_height = self._get_image_dimensions(image_bytes, img_format)
        if actual_width and actual_height:
            width = actual_width
            height = actual_height
        
        # Check minimum size
        if width and height:
            if not self._passes_size_filter(width, height):
                LOGGER.debug("Skipping small image (%dx%d): %s", width, height, src)
                return None
        
        # Generate image ID
        image_id = f"{document_id}_html_img{index + 1}"
        
        # Classify image type based on context
        image_type = self._classify_image(img_tag, alt_text, src)
        
        return ExtractedImage(
            image_id=image_id,
            page_number=1,  # HTML is single "page"
            format=img_format,
            image_bytes=image_bytes,
            bounding_box=None,  # HTML doesn't have fixed positions
            description=None,  # Will be filled by Vision LLM
            image_type=image_type,
            width=width,
            height=height,
            alt_text=alt_text,
            source_url=src if src.startswith(('http://', 'https://')) else None,
        )
    
    def _load_image(
        self,
        src: str,
        asset_root: str,
        base_url: Optional[str],
        html_path: str,
    ) -> tuple[Optional[bytes], str]:
        """
        Load image bytes from various source types.
        
        Returns:
            Tuple of (image_bytes, format) or (None, '') if failed
        """
        # Handle data URLs (base64 embedded)
        if src.startswith('data:'):
            return self._load_data_url(src)
        
        # Handle remote URLs
        if src.startswith(('http://', 'https://')):
            if self.fetch_remote:
                return self._load_remote_url(src)
            else:
                LOGGER.debug("Skipping remote URL (fetch_remote=False): %s", src)
                return None, ''
        
        # Handle relative/absolute file paths
        return self._load_local_file(src, asset_root, html_path)
    
    def _load_data_url(self, data_url: str) -> tuple[Optional[bytes], str]:
        """Load image from a data: URL (base64 encoded)."""
        try:
            # Format: data:image/png;base64,iVBORw0KGgo...
            match = re.match(r'data:image/(\w+);base64,(.+)', data_url, re.IGNORECASE)
            if not match:
                return None, ''
            
            img_format = match.group(1).lower()
            if img_format == 'jpg':
                img_format = 'jpeg'
            
            base64_data = match.group(2)
            image_bytes = base64.b64decode(base64_data)
            
            return image_bytes, img_format
        except Exception as e:
            LOGGER.warning("Failed to decode data URL: %s", e)
            return None, ''
    
    def _load_remote_url(self, url: str) -> tuple[Optional[bytes], str]:
        """Load image from a remote URL."""
        try:
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; ImageExtractor/1.0)'}
            )
            with urllib.request.urlopen(request, timeout=self.remote_timeout) as response:
                image_bytes = response.read()
                content_type = response.headers.get('Content-Type', '')
                
                # Determine format from content type or URL
                img_format = self.MIME_TO_EXT.get(content_type.split(';')[0], '')
                if not img_format:
                    img_format = self._format_from_url(url)
                
                return image_bytes, img_format or 'png'
        except Exception as e:
            LOGGER.warning("Failed to fetch remote image %s: %s", url, e)
            return None, ''
    
    def _load_local_file(
        self,
        src: str,
        asset_root: str,
        html_path: str,
    ) -> tuple[Optional[bytes], str]:
        """Load image from a local file path."""
        # Try multiple resolution strategies
        candidates = []
        
        # 1. Relative to asset_root
        candidates.append(os.path.join(asset_root, src))
        
        # 2. Relative to HTML file location
        html_dir = os.path.dirname(html_path)
        candidates.append(os.path.join(html_dir, src))
        
        # 3. Absolute path (if src starts with /)
        if src.startswith('/'):
            candidates.append(src)
        
        # 4. Try with/without leading slash
        if src.startswith('/'):
            candidates.append(os.path.join(asset_root, src[1:]))
        else:
            candidates.append(os.path.join(asset_root, '/', src))
        
        # 5. Common asset folder names
        for folder in ['images', 'img', 'assets', 'static', 'media']:
            candidates.append(os.path.join(asset_root, folder, os.path.basename(src)))
        
        # Try each candidate
        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                try:
                    with open(candidate, 'rb') as f:
                        image_bytes = f.read()
                    img_format = self._format_from_path(candidate)
                    LOGGER.debug("Loaded image from: %s", candidate)
                    return image_bytes, img_format
                except Exception as e:
                    LOGGER.debug("Failed to read %s: %s", candidate, e)
                    continue
        
        LOGGER.debug("Could not resolve local image path: %s", src)
        return None, ''
    
    def _format_from_path(self, path: str) -> str:
        """Extract image format from file path."""
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if ext == 'jpg':
            return 'jpeg'
        return ext if ext in self.SUPPORTED_FORMATS else 'png'
    
    def _format_from_url(self, url: str) -> str:
        """Extract image format from URL."""
        parsed = urlparse(url)
        return self._format_from_path(parsed.path)
    
    def _extract_url_from_srcset(self, srcset: str) -> Optional[str]:
        """Extract the best URL from an srcset attribute.

        Delegates to the shared :func:`~rag_lib.adapters.markdown.html_utils.extract_url_from_srcset`.
        """
        from rag_lib.adapters.markdown.html_utils import extract_url_from_srcset
        return extract_url_from_srcset(srcset)
    
    def _parse_dimension(self, value: Optional[str]) -> Optional[int]:
        """Parse width/height attribute value to integer."""
        if not value:
            return None
        try:
            # Handle values like "100", "100px", "100%"
            clean = re.sub(r'[^\d]', '', str(value))
            return int(clean) if clean else None
        except (ValueError, TypeError):
            return None
    
    def _get_image_dimensions(
        self,
        image_bytes: bytes,
        img_format: str,
    ) -> tuple[Optional[int], Optional[int]]:
        """Get actual image dimensions from bytes."""
        try:
            # Try PIL/Pillow
            from PIL import Image
            from io import BytesIO
            
            img = Image.open(BytesIO(image_bytes))
            return img.width, img.height
        except ImportError:
            pass
        except Exception as e:
            LOGGER.debug("Could not get image dimensions: %s", e)
        
        return None, None
    
    def _convert_svg_to_png(
        self,
        svg_bytes: bytes,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ) -> tuple[Optional[bytes], str]:
        """
        Convert SVG image bytes to PNG for Vision LLM compatibility.
        
        Args:
            svg_bytes: Raw SVG file bytes
            output_width: Optional output width (maintains aspect ratio if only one specified)
            output_height: Optional output height
            
        Returns:
            Tuple of (png_bytes, 'png') or (None, 'svg') if conversion fails
        """
        if not CAIROSVG_AVAILABLE:
            LOGGER.debug("cairosvg not available for SVG conversion")
            return None, 'svg'
        
        try:
            # Convert SVG to PNG using cairosvg
            png_bytes = cairosvg.svg2png(
                bytestring=svg_bytes,
                output_width=output_width,
                output_height=output_height,
            )
            return png_bytes, 'png'
        except Exception as e:
            LOGGER.warning("Failed to convert SVG to PNG: %s", e)
            return None, 'svg'
    
    def _classify_image(
        self,
        img_tag: Any,
        alt_text: str,
        src: str,
    ) -> ImageType:
        """Classify the image type based on context clues."""
        # Check alt text and src for hints
        text_lower = (alt_text + ' ' + src).lower()
        
        # Common patterns
        if any(word in text_lower for word in ['chart', 'graph', 'plot', 'bar', 'pie', 'line']):
            return ImageType.CHART
        if any(word in text_lower for word in ['table', 'grid', 'data']):
            return ImageType.TABLE
        if any(word in text_lower for word in ['diagram', 'flow', 'architecture', 'schema']):
            return ImageType.DIAGRAM
        if any(word in text_lower for word in ['logo', 'icon', 'brand']):
            return ImageType.LOGO
        if any(word in text_lower for word in ['screenshot', 'screen', 'capture']):
            return ImageType.SCREENSHOT
        if any(word in text_lower for word in ['photo', 'picture', 'image']):
            return ImageType.PHOTO
        if any(word in text_lower for word in ['infographic', 'info', 'stat']):
            return ImageType.INFOGRAPHIC
        
        # Check parent elements for context
        parent = img_tag.parent
        if parent:
            parent_class = ' '.join(parent.get('class', []))
            if 'chart' in parent_class or 'graph' in parent_class:
                return ImageType.CHART
            if 'logo' in parent_class or 'header' in parent_class:
                return ImageType.LOGO
        
        return ImageType.UNKNOWN
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID from file path."""
        path = Path(file_path)
        name = path.stem
        
        # Add hash for uniqueness
        hash_input = f"{file_path}_{os.path.getmtime(file_path)}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        # Clean name
        clean_name = re.sub(r'[^\w\-]', '_', name)
        return f"{clean_name}_{short_hash}"
