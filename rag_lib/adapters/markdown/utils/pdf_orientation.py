"""PDF Page Orientation Detection and Correction Utilities.

This module provides utilities for detecting and correcting page orientation
issues in PDF documents, including:

1. **Rotation metadata detection** - Reads PDF page rotation attribute (0°, 90°, 180°, 270°)
2. **Text-based orientation detection** - Analyzes text direction to detect misorientation
3. **Automatic correction** - Normalizes pages to standard reading orientation

Usage:
    from rag_lib.adapters.markdown.utils.pdf_orientation import (
        PDFOrientationCorrector,
        detect_page_orientation,
        correct_pdf_orientation,
    )
    
    # Correct entire PDF
    corrector = PDFOrientationCorrector()
    corrected_path = corrector.correct_file(pdf_path)
    
    # Or detect orientation of a single page
    rotation = detect_page_orientation(pdf_path, page_num=0)
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NumberObject
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False

# Legacy alias used by callers
PYMUPDF_AVAILABLE = PYPDF_AVAILABLE


def detect_page_rotation_metadata_from_reader(reader_page) -> int:
    """
    Get the rotation angle from a pypdf page object.

    Args:
        reader_page: pypdf PageObject

    Returns:
        Rotation angle in degrees (0, 90, 180, or 270)
    """
    return int(reader_page.get("/Rotate") or 0) % 360


def detect_text_orientation_plumber(plumber_page, sample_lines: int = 40) -> int:
    """
    Detect page orientation by analysing character bounding-box directions
    using *pdfplumber*.

    Strategy: group characters into lines by proximity, then check whether
    lines are predominantly horizontal or vertical.

    Args:
        plumber_page: An open pdfplumber page
        sample_lines: Max character groups to analyse

    Returns:
        Detected rotation needed to correct orientation (0, 90, 180, 270)
    """
    words = plumber_page.extract_words() or []
    if not words:
        return 0

    horizontal_count = 0
    vertical_count = 0

    for word in words[:sample_lines]:
        width = word["x1"] - word["x0"]
        height = word["bottom"] - word["top"]
        if width > height * 1.5:
            horizontal_count += 1
        elif height > width * 1.5:
            vertical_count += 1

    if vertical_count > horizontal_count * 2:
        LOGGER.debug(
            "Text orientation analysis: %d horizontal, %d vertical words – suggesting rotation",
            horizontal_count, vertical_count,
        )
        return 90

    return 0


def analyze_page_for_rotation(file_path_or_bytes, page_idx: int = 0) -> Tuple[int, str]:
    """
    Comprehensive page rotation analysis combining metadata and text analysis.

    Args:
        file_path_or_bytes: PDF file path (str/Path) or in-memory bytes
        page_idx: 0-based page index

    Returns:
        Tuple of (rotation_needed, detection_method)
    """
    source = io.BytesIO(file_path_or_bytes) if isinstance(file_path_or_bytes, bytes) else file_path_or_bytes

    # Method 1: Check PDF rotation metadata via pypdf
    if PYPDF_AVAILABLE:
        try:
            reader = PdfReader(source)
            if page_idx < len(reader.pages):
                rotation = detect_page_rotation_metadata_from_reader(reader.pages[page_idx])
                if rotation != 0:
                    return rotation, "pdf_metadata"
        except Exception as exc:
            LOGGER.debug("pypdf metadata check failed: %s", exc)

    # Reset stream position if bytes
    if isinstance(file_path_or_bytes, bytes):
        source = io.BytesIO(file_path_or_bytes)

    # Method 2: Analyse text orientation via pdfplumber
    if PDFPLUMBER_AVAILABLE:
        try:
            pdf = pdfplumber.open(source)
            if page_idx < len(pdf.pages):
                text_rotation = detect_text_orientation_plumber(pdf.pages[page_idx])
                pdf.close()
                if text_rotation != 0:
                    return text_rotation, "text_analysis"
            else:
                pdf.close()
        except Exception as exc:
            LOGGER.debug("pdfplumber text analysis failed: %s", exc)

    return 0, "none_needed"


class PDFOrientationCorrector:
    """
    PDF orientation detection and correction utility.
    
    This class handles:
    1. Reading PDF rotation metadata (via pypdf)
    2. Detecting misoriented pages via text analysis (via pdfplumber)
    3. Correcting rotation by creating normalized PDFs (via pypdf)
    4. Rendering corrected pages as images (via pypdfium2)
    """
    
    def __init__(
        self,
        auto_detect: bool = True,
        correct_metadata_rotation: bool = True,
        correct_text_rotation: bool = True,
        temp_dir: Optional[str] = None,
    ):
        if not PYPDF_AVAILABLE:
            raise ImportError(
                "pypdf is required for PDF orientation correction. "
                "Install with: pip install pypdf"
            )
        
        self.auto_detect = auto_detect
        self.correct_metadata_rotation = correct_metadata_rotation
        self.correct_text_rotation = correct_text_rotation
        self.temp_dir = temp_dir
        
        LOGGER.info(
            "PDFOrientationCorrector initialized: auto_detect=%s, "
            "metadata_rotation=%s, text_rotation=%s",
            auto_detect, correct_metadata_rotation, correct_text_rotation
        )
    
    def analyze_document(self, file_path: str) -> List[dict]:
        """
        Analyze all pages in a PDF for rotation issues.
        
        Returns:
            List of dicts with page analysis results.
        """
        results = []
        reader = PdfReader(file_path)
        
        for page_num in range(len(reader.pages)):
            rotation_needed, method = analyze_page_for_rotation(file_path, page_num)
            current_rotation = int(reader.pages[page_num].get("/Rotate") or 0) % 360
            
            results.append({
                "page_num": page_num + 1,
                "current_rotation": current_rotation,
                "needs_correction": rotation_needed != 0,
                "correction_angle": rotation_needed,
                "detection_method": method,
            })
            
            if rotation_needed != 0:
                LOGGER.debug(
                    "Page %d needs %d° rotation correction (detected via %s)",
                    page_num + 1, rotation_needed, method
                )
        
        return results
    
    def correct_page(
        self,
        file_path_or_bytes,
        page_num: int,
        rotation_override: Optional[int] = None,
        dpi: int = 150,
    ) -> bytes:
        """
        Correct a single page's orientation and return as PNG image bytes.
        
        Args:
            file_path_or_bytes: PDF path or bytes
            page_num: Page index (0-based)
            rotation_override: Force specific rotation instead of auto-detect
            dpi: Rendering resolution
            
        Returns:
            PNG image bytes of the corrected page
        """
        if rotation_override is not None:
            rotation = rotation_override
        elif self.auto_detect:
            rotation, _ = analyze_page_for_rotation(file_path_or_bytes, page_num)
        else:
            if PYPDF_AVAILABLE:
                source = io.BytesIO(file_path_or_bytes) if isinstance(file_path_or_bytes, bytes) else file_path_or_bytes
                reader = PdfReader(source)
                rotation = int(reader.pages[page_num].get("/Rotate") or 0) % 360 if self.correct_metadata_rotation else 0
            else:
                rotation = 0
        
        if not PYPDFIUM2_AVAILABLE:
            raise ImportError("pypdfium2 is required for page rendering")
        
        source = io.BytesIO(file_path_or_bytes) if isinstance(file_path_or_bytes, bytes) else file_path_or_bytes
        pdf = pdfium.PdfDocument(source)
        page = pdf[page_num]
        
        scale = dpi / 72
        # Apply counter-rotation: pypdfium2 rotation is CCW in degrees (0, 90, 180, 270)
        render_rotation = (-rotation) % 360 if rotation != 0 else 0
        
        bitmap = page.render(scale=scale, rotation=render_rotation)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        pdf.close()
        
        if rotation != 0:
            LOGGER.debug("Applied %d° rotation correction to page %d", rotation, page_num + 1)
        
        return png_bytes
    
    def correct_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        pages: Optional[List[int]] = None,
    ) -> str:
        """
        Correct page orientations in a PDF file.
        
        Creates a new PDF with all pages normalized to correct orientation.
        """
        if output_path is None:
            temp_dir = self.temp_dir or tempfile.gettempdir()
            base_name = Path(file_path).stem
            output_path = os.path.join(temp_dir, f"{base_name}_corrected.pdf")
        
        reader = PdfReader(file_path)
        writer = PdfWriter()
        corrections_made = 0
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            writer.add_page(page)
            
            if pages is not None and (page_num + 1) not in pages:
                continue
            
            if self.auto_detect:
                rotation_needed, method = analyze_page_for_rotation(file_path, page_num)
            else:
                current = int(page.get("/Rotate") or 0) % 360
                rotation_needed = current if self.correct_metadata_rotation else 0
                method = "metadata" if rotation_needed else "none"
            
            if rotation_needed != 0:
                writer.pages[page_num].update({"/Rotate": NumberObject(0)})
                LOGGER.info(
                    "Corrected page %d: removed %d° rotation (detected via %s)",
                    page_num + 1, rotation_needed, method
                )
                corrections_made += 1
        
        with open(output_path, "wb") as f:
            writer.write(f)
        
        if corrections_made > 0:
            LOGGER.info(
                "Saved corrected PDF with %d page rotations fixed: %s",
                corrections_made, output_path
            )
        else:
            LOGGER.debug("No rotation corrections needed, saved copy: %s", output_path)
        
        return output_path
    
    def correct_single_page_pdf(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Correct orientation of a single-page PDF.
        """
        reader = PdfReader(file_path)
        
        if len(reader.pages) != 1:
            LOGGER.warning(
                "correct_single_page_pdf called on PDF with %d pages",
                len(reader.pages)
            )
        
        rotation_needed, method = analyze_page_for_rotation(file_path, 0)
        
        if rotation_needed != 0:
            writer = PdfWriter()
            writer.add_page(reader.pages[0])
            writer.pages[0].update({"/Rotate": NumberObject(0)})
            
            save_path = output_path or file_path
            with open(save_path, "wb") as f:
                writer.write(f)
            
            LOGGER.debug(
                "Corrected single-page PDF rotation: %d° (via %s) -> %s",
                rotation_needed, method, save_path
            )
            return save_path
        
        if output_path and output_path != file_path:
            writer = PdfWriter()
            writer.add_page(reader.pages[0])
            with open(output_path, "wb") as f:
                writer.write(f)
            return output_path
        
        return file_path


def detect_page_orientation(file_path: str, page_num: int = 0) -> int:
    """
    Convenience function to detect rotation of a specific page.
    """
    if not PYPDF_AVAILABLE:
        LOGGER.warning("pypdf not available for orientation detection")
        return 0
    
    rotation, _ = analyze_page_for_rotation(file_path, page_num)
    return rotation


def correct_pdf_orientation(
    file_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convenience function to correct PDF orientation.
    """
    corrector = PDFOrientationCorrector()
    return corrector.correct_file(file_path, output_path)


def needs_orientation_correction(file_path: str) -> bool:
    """
    Quick check if a PDF has any pages needing rotation correction.
    """
    if not PYPDF_AVAILABLE:
        return False
    
    reader = PdfReader(file_path)
    for page_num in range(len(reader.pages)):
        rotation, _ = analyze_page_for_rotation(file_path, page_num)
        if rotation != 0:
            return True
    return False
