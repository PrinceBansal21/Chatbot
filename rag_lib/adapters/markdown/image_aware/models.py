"""
Data models for image-aware markdown conversion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class ImageType(Enum):
    """Type of extracted image."""
    CHART = "chart"
    TABLE = "table"
    DIAGRAM = "diagram"
    INFOGRAPHIC = "infographic"
    PHOTO = "photo"
    LOGO = "logo"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates for an image."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class ExtractedImage:
    """Represents an extracted image from a document."""
    
    # Core identification
    image_id: str
    page_number: int
    
    # Image data
    image_bytes: bytes
    format: str = "png"  # png, jpg, etc.
    
    # Location
    bounding_box: Optional[BoundingBox] = None
    
    # Metadata
    description: Optional[str] = None
    image_type: ImageType = ImageType.UNKNOWN
    
    # Storage
    storage_path: Optional[str] = None  # Path where image is stored
    storage_url: Optional[str] = None   # URL for accessing the image
    
    # Additional metadata
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    
    # HTML-specific metadata
    alt_text: Optional[str] = None      # Alt text from <img> tag
    source_url: Optional[str] = None    # Original URL for remote images
    
    # Infographic classification (set by Vision LLM during description)
    is_infographic: bool = False
    
    # Decorative classification (set by Vision LLM during description)
    # True = logos, icons, navigation bars, banners, watermarks, etc.
    is_decorative: bool = False
    
    # Page dimensions (PDF points) — enables spatial placement in markdown
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    
    # Orientation correction applied (degrees clockwise: 0, 90, 180, 270)
    rotation_corrected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_id": self.image_id,
            "page_number": self.page_number,
            "format": self.format,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "description": self.description,
            "image_type": self.image_type.value,
            "storage_path": self.storage_path,
            "storage_url": self.storage_url,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "alt_text": self.alt_text,
            "source_url": self.source_url,
            "is_infographic": self.is_infographic,
            "is_decorative": self.is_decorative,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "rotation_corrected": self.rotation_corrected,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], image_bytes: bytes = b"") -> "ExtractedImage":
        """Create from dictionary."""
        bbox = None
        if data.get("bounding_box"):
            bbox = BoundingBox(**data["bounding_box"])
        
        return cls(
            image_id=data["image_id"],
            page_number=data["page_number"],
            image_bytes=image_bytes,
            format=data.get("format", "png"),
            bounding_box=bbox,
            description=data.get("description"),
            image_type=ImageType(data.get("image_type", "unknown")),
            storage_path=data.get("storage_path"),
            storage_url=data.get("storage_url"),
            width=data.get("width"),
            height=data.get("height"),
            file_size=data.get("file_size"),
            alt_text=data.get("alt_text"),
            source_url=data.get("source_url"),
            is_infographic=data.get("is_infographic", False),
            is_decorative=data.get("is_decorative", False),
            page_width=data.get("page_width"),
            page_height=data.get("page_height"),
            rotation_corrected=data.get("rotation_corrected", 0),
        )


@dataclass
class ImageManifest:
    """Manifest containing all images from a document."""
    
    document_id: str
    source_file: str
    images: List[ExtractedImage] = field(default_factory=list)
    total_pages: int = 0
    extraction_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "total_pages": self.total_pages,
            "extraction_timestamp": self.extraction_timestamp,
            "image_count": len(self.images),
            "images": [img.to_dict() for img in self.images],
        }
    
    def get_images_for_page(self, page_number: int) -> List[ExtractedImage]:
        """Get all images for a specific page."""
        return [img for img in self.images if img.page_number == page_number]


@dataclass
class MarkdownWithImages:
    """Result of image-aware markdown conversion."""
    
    markdown: str
    images: ImageManifest
    
    # Metadata
    document_id: str
    source_file: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "markdown": self.markdown,
            "images": self.images.to_dict(),
        }
