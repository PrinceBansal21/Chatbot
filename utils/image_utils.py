"""Image-related utilities for configuration resolution, chunk linking, and markdown preprocessing."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_lib.interfaces.chunking_manager import Chunk

from .logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


# =============================================================================
# IMAGE CONFIGURATION RESOLVER
# =============================================================================

class ImageConfigResolver:
    """
    Resolves image-related configuration from multiple sources.
    
    Priority order: payload override > file-type config > global config
    """
    
    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the image config resolver.
        
        Args:
            global_config: Global application configuration
        """
        self.global_config = global_config
        self.image_cfg = global_config.get("image_extraction", {})
    
    def should_create_image_chunks(
        self, 
        payload: Dict, 
        file_type_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if dedicated image chunks should be created.
        
        Args:
            payload: Request payload (may contain override settings)
            file_type_config: File-type-specific configuration
            
        Returns:
            True if image chunks should be created
        """
        # Check payload override first
        payload_image_cfg = payload.get("image_extraction", {})
        if "create_image_chunks" in payload_image_cfg:
            return payload_image_cfg["create_image_chunks"]
        
        # Check file-type config
        if file_type_config:
            ft_image_cfg = file_type_config.get("image_extraction", {})
            if "create_image_chunks" in ft_image_cfg:
                return ft_image_cfg["create_image_chunks"]
        
        # Check global config - default to True when image extraction is enabled
        return self.image_cfg.get("create_image_chunks", True)
    
    def should_link_images_to_chunks(
        self, 
        payload: Dict,
        file_type_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if images should be linked to text chunks.
        
        When enabled, text chunks will include metadata about images on the same page,
        allowing the chatbot to display relevant images when the chunk is cited.
        
        Args:
            payload: Request payload (may contain override settings)
            file_type_config: File-type-specific configuration
            
        Returns:
            True if images should be linked to text chunks
        """
        # Check payload override
        payload_image_cfg = payload.get("image_extraction", {})
        if "link_images_to_chunks" in payload_image_cfg:
            return payload_image_cfg["link_images_to_chunks"]
        
        # Check file-type config
        if file_type_config:
            ft_image_cfg = file_type_config.get("image_extraction", {})
            if "link_images_to_chunks" in ft_image_cfg:
                return ft_image_cfg["link_images_to_chunks"]
        
        # Check global config - default to True when image extraction is enabled
        return self.image_cfg.get("link_images_to_chunks", self.image_cfg.get("enabled", False))
    
    def get_max_images_per_chunk(
        self, 
        payload: Dict,
        file_type_config: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get the maximum number of images to link per chunk.
        
        Args:
            payload: Request payload (may contain override settings)
            file_type_config: File-type-specific configuration
            
        Returns:
            Maximum images per chunk (default: 5)
        """
        # Check payload override
        payload_image_cfg = payload.get("image_extraction", {})
        if "max_images_per_chunk" in payload_image_cfg:
            return int(payload_image_cfg["max_images_per_chunk"])
        
        # Check file-type config
        if file_type_config:
            ft_image_cfg = file_type_config.get("image_extraction", {})
            if "max_images_per_chunk" in ft_image_cfg:
                return int(ft_image_cfg["max_images_per_chunk"])
        
        # Check global config
        return self.image_cfg.get("max_images_per_chunk", 5)
    
    def should_embed_in_markdown(self) -> bool:
        """Check if images should be embedded in markdown."""
        return self.image_cfg.get("embed_in_markdown", True)
    
    def is_extraction_enabled(
        self,
        file_type_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if image extraction is enabled.
        
        Args:
            file_type_config: File-type-specific configuration
            
        Returns:
            True if image extraction is enabled
        """
        global_enabled = self.image_cfg.get("enabled", False)
        
        if file_type_config:
            ft_image_cfg = file_type_config.get("image_extraction", {})
            # If file-type config explicitly sets enabled, use that
            if "enabled" in ft_image_cfg:
                return ft_image_cfg["enabled"]
        
        return global_enabled


# =============================================================================
# MARKDOWN PREPROCESSOR
# =============================================================================

class MarkdownPreprocessor:
    """Preprocesses markdown content before chunking."""
    
    # Pattern to match markdown image syntax: ![alt text](url) or ![alt text](url "title")
    IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    
    # Pattern to match HTML comment image placeholders:
    #   <!-- image -->                              (Docling bare placeholder)
    #   <!-- image: abc-123 (page 2) desc -->       (image_aware adapter)
    #   <!-- Images on this page -->                (image_aware section marker)
    IMAGE_COMMENT_PATTERN = re.compile(
        r'<!--\s*[Ii]mage[^>]*-->', re.DOTALL
    )
    
    # Pattern to clean up multiple consecutive newlines
    NEWLINE_PATTERN = re.compile(r'\n{3,}')
    
    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the markdown preprocessor.
        
        Args:
            global_config: Global application configuration
        """
        self.global_config = global_config
    
    def prepare_for_chunking(
        self,
        markdown: str,
        payload: Dict,
        file_type_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Prepare markdown for text chunking by optionally removing image references.
        
        When image chunks are created separately, removing image references from text chunks
        avoids redundancy and keeps text chunks focused on textual content.
        
        Args:
            markdown: Original markdown content
            payload: Request payload (may contain override settings)
            file_type_config: File-type-specific configuration
            
        Returns:
            Processed markdown (with or without image references)
        """
        should_strip = self._should_strip_image_references(payload, file_type_config)
        
        if not should_strip:
            return markdown
        
        return self.strip_image_references(markdown)
    
    def strip_image_references(self, markdown: str) -> str:
        """
        Remove all image references from markdown, including:
        - Markdown image syntax: ![alt](url)
        - HTML comment image placeholders: <!-- image -->, <!-- image: ... -->
        
        Args:
            markdown: Markdown content with image references
            
        Returns:
            Markdown with image references removed
        """
        cleaned_markdown = self.IMAGE_PATTERN.sub('', markdown)
        cleaned_markdown = self.IMAGE_COMMENT_PATTERN.sub('', cleaned_markdown)
        
        # Clean up multiple consecutive newlines that may result from removal
        cleaned_markdown = self.NEWLINE_PATTERN.sub('\n\n', cleaned_markdown)
        
        # Log if images were removed
        original_len = len(markdown)
        cleaned_len = len(cleaned_markdown)
        if cleaned_len < original_len:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                f"Stripped image references from markdown ({original_len} -> {cleaned_len} chars)"
            )
        
        return cleaned_markdown
    
    def _should_strip_image_references(
        self,
        payload: Dict,
        file_type_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine if image references should be stripped from markdown.
        
        Priority: payload > file-type config > global config
        
        Args:
            payload: Request payload
            file_type_config: File-type-specific configuration
            
        Returns:
            True if image references should be stripped
        """
        # Check payload override
        payload_chunking = payload.get("chunking", {})
        if "strip_image_references" in payload_chunking:
            return payload_chunking["strip_image_references"]
        
        # Check file-type config
        if file_type_config:
            chunking_config = file_type_config.get("chunking", {})
            if "strip_image_references" in chunking_config:
                return chunking_config["strip_image_references"]
        
        # Check global config - default to True
        return self.global_config.get("chunking", {}).get("strip_image_references", True)


# =============================================================================
# METADATA ENRICHER
# =============================================================================

class MetadataEnricher:
    """Enriches metadata with image-related information."""
    
    @staticmethod
    def add_image_metadata(
        metadata: Dict[str, str],
        image_manifest: Dict[str, Any],
    ) -> None:
        """
        Update metadata with image extraction information.
        
        Args:
            metadata: Document metadata to update (modified in place)
            image_manifest: Image extraction manifest
        """
        metadata["image_count"] = str(image_manifest.get("image_count", 0))
        metadata["has_images"] = "true" if image_manifest.get("image_count", 0) > 0 else "false"
        if image_manifest.get("manifest_location"):
            metadata["image_manifest_location"] = image_manifest["manifest_location"]
    
    @staticmethod
    def extract_field_names(config_section: Dict) -> list:
        """
        Extract field names from a config section, excluding description keys.
        
        Args:
            config_section: Configuration section with field definitions
            
        Returns:
            List of field names (excluding keys starting with '_')
        """
        if not config_section:
            return []
        return [k for k in config_section.keys() if not k.startswith("_")]


# =============================================================================
# IMAGE CHUNK LINKER
# =============================================================================

class ImageChunkLinker:
    """
    Links images from manifest to text chunks based on page numbers.
    
    When a text chunk is retrieved during RAG, the linked images can be
    displayed alongside the answer if they contributed to the response.
    """
    
    # Maximum images to link per chunk (to avoid bloating metadata)
    MAX_IMAGES_PER_CHUNK = 5
    
    # Fields to include in chunk metadata for each linked image
    IMAGE_METADATA_FIELDS = [
        "image_id", "storage_path", "description", "page_number",
        "image_type", "width", "height"
    ]
    
    @classmethod
    def link_images_to_chunks(
        cls,
        chunks: List[Chunk],
        image_manifest: Optional[Dict[str, Any]],
        max_images_per_chunk: int = MAX_IMAGES_PER_CHUNK,
    ) -> List[Chunk]:
        """
        Link images from manifest to text chunks based on page numbers.
        
        Args:
            chunks: List of text chunks to enrich
            image_manifest: Image manifest dictionary with 'images' list
            max_images_per_chunk: Maximum images to link per chunk
            
        Returns:
            Chunks with linked image metadata added
        """
        if not image_manifest:
            return chunks
        
        images = image_manifest.get("images", [])
        if not images:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                "No images in manifest to link to chunks"
            )
            return chunks
        
        # Build page -> images mapping
        images_by_page = cls._build_page_image_map(images)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Linking images to chunks: {len(images)} images across {len(images_by_page)} pages"
        )
        
        linked_count = 0
        
        for chunk in chunks:
            # Skip non-text chunks (e.g., image chunks, summary chunks)
            chunk_type = chunk.metadata.get("chunk_type", "text")
            if chunk_type != "text":
                continue
            
            # Get page number from chunk
            page_numbers = cls._extract_page_numbers(chunk)
            
            if not page_numbers:
                continue
            
            # Collect images for this chunk's pages
            chunk_images = []
            for page_num in page_numbers:
                if page_num in images_by_page:
                    chunk_images.extend(images_by_page[page_num])
            
            if not chunk_images:
                continue
            
            # Limit number of images
            chunk_images = chunk_images[:max_images_per_chunk]
            
            # Add image metadata to chunk
            cls._add_image_metadata_to_chunk(chunk, chunk_images)
            linked_count += 1
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Linked images to {linked_count} text chunks"
        )
        
        return chunks
    
    @classmethod
    def _build_page_image_map(cls, images: List[Dict]) -> Dict[int, List[Dict]]:
        """Build a mapping from page number to list of images."""
        page_map: Dict[int, List[Dict]] = {}
        
        for img in images:
            page_num = cls._get_image_page(img)
            if page_num is not None:
                if page_num not in page_map:
                    page_map[page_num] = []
                page_map[page_num].append(img)
        
        return page_map
    
    @classmethod
    def _get_image_page(cls, img: Dict) -> Optional[int]:
        """Extract page number from image dict or object."""
        page = img.get("page_number") or img.get("page")
        if page is not None:
            try:
                return int(page)
            except (ValueError, TypeError):
                pass
        return None
    
    @classmethod
    def _extract_page_numbers(cls, chunk: Chunk) -> List[int]:
        """
        Extract page numbers from a chunk (from metadata or content).
        
        Returns:
            List of page numbers associated with this chunk
        """
        page_numbers = set()
        
        # Check metadata first
        page_meta = chunk.metadata.get("page_number") or chunk.metadata.get("page")
        if page_meta:
            try:
                page_numbers.add(int(page_meta))
            except (ValueError, TypeError):
                pass
        
        # Also parse content for page markers
        # Pattern: <!-- page_number: 5 -->
        pattern = r'<!--\s*page_number:\s*(\d+)\s*-->'
        matches = re.findall(pattern, chunk.content, re.IGNORECASE)
        for match in matches:
            try:
                page_numbers.add(int(match))
            except ValueError:
                pass
        
        return sorted(list(page_numbers))
    
    @classmethod
    def _add_image_metadata_to_chunk(cls, chunk: Chunk, images: List[Dict]) -> None:
        """
        Add linked media metadata to chunk under a unified ``media`` key.

        Builds a JSON object with separate arrays for infographics, images,
        and videos (future).  Also sets fast-filter booleans:
        - ``has_media``: "true" if any media is linked
        - ``infographic``: "true" if at least one linked item is an infographic
        """
        infographics: List[Dict[str, str]] = []
        regular_images: List[Dict[str, str]] = []

        for img in images:
            entry: Dict[str, str] = {}

            # Core identifiers
            for field in ("image_id",):
                value = img.get(field)
                if value is not None:
                    entry["id"] = str(value)

            # URL — try multiple possible keys
            storage_path = (
                img.get("storage_path")
                or img.get("storage_url")
            )
            if storage_path:
                entry["url"] = storage_path

            # Description
            desc = img.get("description")
            if desc:
                entry["description"] = str(desc)[:500]

            # Page
            page = img.get("page_number")
            if page is not None:
                entry["page"] = str(page)

            # Dimensions
            for dim in ("width", "height"):
                val = img.get(dim)
                if val is not None:
                    entry[dim] = str(val)

            # Relevance (from Vision LLM tagging)
            relevance = img.get("relevance")
            if relevance:
                entry["relevance"] = relevance

            # Image type / format
            img_type = img.get("image_type")
            if img_type:
                entry["image_type"] = img_type

            if not entry:
                continue

            # Route to the right array
            is_infographic = bool(img.get("is_infographic", False))
            if is_infographic:
                infographics.append(entry)
            else:
                regular_images.append(entry)

        # Build the media envelope
        media = {
            "infographics": infographics,
            "images": regular_images,
            "videos": [],
        }

        chunk.metadata["media"] = json.dumps(media)
        chunk.metadata["has_media"] = "true"

        # Fast-filter boolean for infographic presence
        if infographics:
            chunk.metadata["infographic"] = "true"


# =============================================================================
# IMAGE RELEVANCE SCORER (cosine similarity)
# =============================================================================

class ImageRelevanceScorer:
    """
    Scores image-chunk relevance using cosine similarity between text chunk
    embeddings and image description embeddings.

    This is a zero-cost second pass — it reuses the text chunk vectors that
    were already computed during the embedding stage and generates a small
    number of additional embeddings for image descriptions only.

    After scoring, each text chunk with linked media gets:
    - ``has_relevant_media``: "true" / "false" (fast vector database filter)
      - ``max_image_relevance_score``: float 0-1, highest cosine sim
      - Per-image ``relevance_score`` inside the ``media`` JSON
    """

    # A chunk's best-matching image must score above this to be "relevant"
    DEFAULT_THRESHOLD = 0.30

    @classmethod
    def score(
        cls,
        embeddings: List[Tuple[Chunk, List[float]]],
        embed_fn,
        relevance_threshold: float = DEFAULT_THRESHOLD,
    ) -> int:
        """
        Score image relevance for all text chunks that have linked media.

        Args:
            embeddings: List of (Chunk, vector) tuples from the embedding stage.
            embed_fn: Callable that takes a list of Chunk objects and returns
                      List[Tuple[Chunk, List[float]]].  Used to embed image
                      descriptions.
            relevance_threshold: Cosine similarity threshold for "relevant".

        Returns:
            Number of chunks that received relevance scores.
        """
        # Collect text chunks that have linked media
        media_chunks: List[Tuple[Chunk, List[float]]] = []
        for chunk, vec in embeddings:
            if chunk.metadata.get("has_media") == "true" and chunk.metadata.get("chunk_type", "text") == "text":
                media_chunks.append((chunk, vec))

        if not media_chunks:
            return 0

        # Gather unique image descriptions from all linked media
        desc_set: Dict[str, str] = {}  # description -> temp_id
        for chunk, _ in media_chunks:
            media_raw = chunk.metadata.get("media")
            if not media_raw:
                continue
            try:
                media_obj = json.loads(media_raw) if isinstance(media_raw, str) else media_raw
            except (json.JSONDecodeError, TypeError):
                continue
            for arr_key in ("images", "infographics"):
                for img in media_obj.get(arr_key, []):
                    desc = img.get("description", "").strip()
                    if desc and desc not in desc_set:
                        desc_set[desc] = f"img-desc-{len(desc_set)}"

        if not desc_set:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                "Image relevance scoring: no image descriptions to score",
            )
            return 0

        # Embed all unique image descriptions in one batch
        desc_chunks = [
            Chunk(chunk_id=tid, file_id="", content=desc, order=0, metadata={})
            for desc, tid in desc_set.items()
        ]
        desc_embeddings = embed_fn(desc_chunks)
        desc_vectors: Dict[str, List[float]] = {}
        for dchunk, dvec in desc_embeddings:
            desc_vectors[dchunk.content.strip()] = dvec

        # Score each media chunk
        scored_count = 0
        for chunk, chunk_vec in media_chunks:
            media_raw = chunk.metadata.get("media")
            if not media_raw:
                continue
            try:
                media_obj = json.loads(media_raw) if isinstance(media_raw, str) else media_raw
            except (json.JSONDecodeError, TypeError):
                continue

            max_score = 0.0
            updated = False
            for arr_key in ("images", "infographics"):
                for img in media_obj.get(arr_key, []):
                    desc = img.get("description", "").strip()
                    if desc and desc in desc_vectors:
                        sim = cls._cosine_similarity(chunk_vec, desc_vectors[desc])
                        img["relevance_score"] = round(sim, 4)
                        img["relevance"] = "relevant" if sim >= relevance_threshold else "unrelated"
                        max_score = max(max_score, sim)
                        updated = True

            if updated:
                chunk.metadata["media"] = json.dumps(media_obj)
                chunk.metadata["max_image_relevance_score"] = str(round(max_score, 4))
                chunk.metadata["has_relevant_media"] = "true" if max_score >= relevance_threshold else "false"
                scored_count += 1

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Image relevance scoring: scored {scored_count} chunks "
            f"({len(desc_set)} unique descriptions, threshold={relevance_threshold})",
        )
        return scored_count

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def link_images_to_chunks(
    chunks: List[Chunk],
    image_manifest: Optional[Dict[str, Any]],
    max_images_per_chunk: int = 5,
) -> List[Chunk]:
    """
    Convenience function to link images to chunks.
    
    Args:
        chunks: List of text chunks to enrich
        image_manifest: Image manifest dictionary
        max_images_per_chunk: Maximum images per chunk
        
    Returns:
        Chunks with linked image metadata
    """
    return ImageChunkLinker.link_images_to_chunks(
        chunks, image_manifest, max_images_per_chunk
    )


__all__ = [
    "ImageConfigResolver",
    "MarkdownPreprocessor", 
    "MetadataEnricher",
    "ImageChunkLinker",
    "ImageRelevanceScorer",
    "link_images_to_chunks",
]
