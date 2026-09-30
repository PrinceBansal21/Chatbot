"""PDF document helpers shared across extractors and adapters."""
from __future__ import annotations


def get_pdf_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF document.

    Uses pypdf to open the file.  Returns ``0`` on any
    error (missing file, non-PDF, pypdf not installed, etc.).
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception:
        return 0
