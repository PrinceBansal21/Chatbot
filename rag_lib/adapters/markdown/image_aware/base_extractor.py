"""Base class for all image extractors.

Provides shared constructor parameters (``min_width``, ``min_height``)
and common helpers so that subclasses don't duplicate boilerplate.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from .models import ExtractedImage

LOGGER = logging.getLogger(__name__)


class BaseImageExtractor:
    """Common base for PDF, HTML, DOCX, and PPTX image extractors.

    Subclasses **must** implement :meth:`extract_images`.
    """

    def __init__(
        self,
        *,
        min_width: int = 50,
        min_height: int = 50,
        **kwargs,
    ):
        self.min_width = min_width
        self.min_height = min_height

    # ------------------------------------------------------------------
    #  Shared helpers
    # ------------------------------------------------------------------

    def _passes_size_filter(self, width: int, height: int) -> bool:
        """Return ``True`` if the image meets the minimum size thresholds."""
        return width >= self.min_width and height >= self.min_height

    def _generate_document_id(self, file_path: str) -> str:
        """Generate a deterministic document ID from *file_path*.

        Subclasses may override for richer hashing (e.g. include mtime).
        """
        path = Path(file_path)
        path_hash = hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:8]
        return f"{path.stem}_{path_hash}"

    # ------------------------------------------------------------------
    #  Abstract contract
    # ------------------------------------------------------------------

    def extract_images(
        self,
        file_path: str,
        *,
        document_id: Optional[str] = None,
        **kwargs,
    ) -> List[ExtractedImage]:
        """Extract images from the given file.

        Must be implemented by every subclass.
        """
        raise NotImplementedError
