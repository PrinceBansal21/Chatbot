from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Dict, List, Tuple, Optional

from rag_lib.utils.page_markers import PAGE_MARKER_RE

@dataclass
class Chunk:
    """Represents a single chunk ready for embedding.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        file_id: ID of the source file
        content: The actual text content of the chunk (stored in vector DB)
        order: Sequential order of this chunk within the document
        metadata: Additional metadata (page_number, headers, etc.)
        parent_id: For hierarchical chunking - ID of the parent chunk (None for top-level)
        level: For hierarchical chunking - 'parent', 'child', or 'summary'
        embedding_content: Optional contextual text used for embedding generation.
            When set, the embedding adapter uses this text to generate the vector
            (e.g. Title + Headers + Page Summary + Chunk), while `content` remains
            the original chunk text stored in the vector DB for retrieval display.
            If None, the embedding adapter falls back to `content`.
    """
    chunk_id: str
    file_id: str
    content: str
    order: int
    metadata: Dict[str, str]
    parent_id: Optional[str] = None
    level: str = "chunk"  # 'parent', 'child', 'summary', or 'chunk' (flat)
    embedding_content: Optional[str] = None
    image_base64: Optional[str] = None  # base64-encoded image for multimodal embedding (embed-v4)

class ChunkingManager(ABC):
    """Interface for document chunking strategies."""
    
    max_tokens: int
    overlap_tokens: int

    @abstractmethod
    def chunk(self, file_id: str, markdown: str, base_metadata: Dict[str, str]) -> List[Chunk]:
        """
        Splits the markdown content into chunks.

        Args:
            file_id: The unique identifier of the file.
            markdown: The markdown content to chunk.
            base_metadata: Metadata to attach to each chunk.

        Returns:
            A list of Chunk objects.
        """
        pass

    # Pattern for HTML comments that are NOT page markers.
    # Matches e.g. <!-- source: docling -->, <!-- source: vision_llm -->,
    # <!-- source: docling+vision -->, and any other non-page_number comments.
    _NON_PAGE_COMMENT_RE = re.compile(
        r"<!--\s*(?!page_number\b)[^>]*?-->\s*", re.IGNORECASE
    )

    @classmethod
    def _strip_non_page_comments(cls, text: str) -> str:
        """Remove HTML comments that are not page markers (e.g. source annotations)."""
        return cls._NON_PAGE_COMMENT_RE.sub("", text)

    @classmethod
    def _split_by_page(cls, markdown: str) -> List[Tuple[str, Optional[str]]]:
        """
        Splits markdown by page markers.
        
        Non-page HTML comments (e.g. ``<!-- source: docling -->``) are
        stripped from each page's content so they never appear in chunks.
        
        Content before the first page marker (preamble) is assigned to
        page "1" as a fallback so that chunks always have a page_number.
        
        Returns list of (content, page_number).
        """
        parts = PAGE_MARKER_RE.split(markdown)
        
        results = []
        # Preamble (before first marker) – assign to page 1 instead of None
        preamble = cls._strip_non_page_comments(parts[0])
        if preamble.strip():
            # If there are subsequent page markers, use the first one;
            # otherwise default to "1".
            first_page = parts[1] if len(parts) > 1 else "1"
            results.append((preamble, str(first_page)))
            
        for i in range(1, len(parts), 2):
            page_num = parts[i]
            content = cls._strip_non_page_comments(parts[i+1])
            if content.strip():
                results.append((content, str(page_num)))
                
        return results
