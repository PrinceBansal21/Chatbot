"""Shared HTML utilities for image extraction.

Provides helpers used by both the docling markdown adapter and the
image-aware HTML image extractor.
"""
from __future__ import annotations

from typing import Optional


def extract_url_from_srcset(srcset: str) -> Optional[str]:
    """Extract the highest-resolution URL from an HTML ``srcset`` attribute.

    Handles both *width descriptors* (``300w``, ``600w``) and
    *density descriptors* (``1x``, ``2x``, ``1.5x``).

    Args:
        srcset: The raw ``srcset`` attribute value.

    Returns:
        Best URL string, or *None* if *srcset* is empty /
        no valid entries found.
    """
    if not srcset:
        return None

    entries = [entry.strip() for entry in srcset.split(",")]

    best_url: Optional[str] = None
    best_size = 0

    for entry in entries:
        if not entry:
            continue

        parts = entry.split()
        if not parts:
            continue

        url = parts[0].strip()
        if not url:
            continue

        size = 1  # default when no descriptor present

        if len(parts) > 1:
            descriptor = parts[-1].lower()

            if descriptor.endswith("w"):
                try:
                    size = int(descriptor[:-1])
                except ValueError:
                    size = 1
            elif descriptor.endswith("x"):
                try:
                    size = int(float(descriptor[:-1]) * 1000)
                except ValueError:
                    size = 1

        if size >= best_size:
            best_size = size
            best_url = url

    return best_url
