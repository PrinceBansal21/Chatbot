"""Markdown adapter utilities.

Common utilities shared across markdown adapters.
"""

from .pdf_orientation import (
    PDFOrientationCorrector,
    detect_page_orientation,
    correct_pdf_orientation,
    needs_orientation_correction,
    analyze_page_for_rotation,
)

__all__ = [
    "PDFOrientationCorrector",
    "detect_page_orientation", 
    "correct_pdf_orientation",
    "needs_orientation_correction",
    "analyze_page_for_rotation",
]
