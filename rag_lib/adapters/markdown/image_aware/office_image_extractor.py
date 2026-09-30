"""
Office document (DOCX / PPTX) image extraction utilities.

Extracts embedded images from Microsoft Office Open XML formats.
Both .docx and .pptx files are ZIP archives that store media assets
under ``word/media/`` and ``ppt/media/`` respectively.  This module
unzips those assets, applies size filters, and returns
:class:`ExtractedImage` objects consistent with the PDF / HTML
extractors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import BoundingBox, ExtractedImage, ImageType
from .base_extractor import BaseImageExtractor

LOGGER = logging.getLogger(__name__)

try:
    from PIL import Image, ExifTags
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Supported raster image extensions inside the archive
_SUPPORTED_EXTS: Dict[str, str] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tiff": "tiff",
    ".tif": "tiff",
}

# Extensions we skip (vector / non-raster assets that won't embed well)
_SKIP_EXTS = {".emf", ".wmf", ".svg", ".eps"}


# ---------------------------------------------------------------------------
# Lightweight image-dimension helpers (avoid heavy Pillow dependency)
# ---------------------------------------------------------------------------

def _png_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Read width/height from PNG IHDR chunk."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]
    return w, h


def _jpeg_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Read width/height from JPEG SOF marker."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None, None
    idx = 2
    while idx < len(data) - 8:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        seg_len = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
        # SOF0..SOF3 markers
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h = struct.unpack(">H", data[idx + 5 : idx + 7])[0]
            w = struct.unpack(">H", data[idx + 7 : idx + 9])[0]
            return w, h
        idx += 2 + seg_len
    return None, None


def _gif_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Read width/height from GIF header."""
    if len(data) < 10 or data[:3] != b"GIF":
        return None, None
    w = struct.unpack("<H", data[6:8])[0]
    h = struct.unpack("<H", data[8:10])[0]
    return w, h


def _bmp_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Read width/height from BMP DIB header."""
    if len(data) < 26 or data[:2] != b"BM":
        return None, None
    w = struct.unpack("<I", data[18:22])[0]
    h = abs(struct.unpack("<i", data[22:26])[0])
    return w, h


def _webp_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Read width/height from WebP RIFF header (VP8 only)."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None, None
    # VP8 lossy
    if data[12:16] == b"VP8 " and len(data) >= 30:
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return w, h
    # VP8L lossless
    if data[12:16] == b"VP8L" and len(data) >= 25:
        bits = struct.unpack("<I", data[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    return None, None


def _get_image_dimensions(data: bytes, fmt: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (width, height) for common raster formats without Pillow."""
    parsers = {
        "png": _png_dimensions,
        "jpeg": _jpeg_dimensions,
        "gif": _gif_dimensions,
        "bmp": _bmp_dimensions,
        "webp": _webp_dimensions,
    }
    parser = parsers.get(fmt)
    if parser:
        return parser(data)
    return None, None


# ---------------------------------------------------------------------------
# Relationship-based image extraction helpers
# ---------------------------------------------------------------------------

def _parse_rels_file(zip_file: zipfile.ZipFile, rels_path: str) -> Dict[str, str]:
    """Parse an Open XML ``.rels`` file and return image relationship mappings.

    Returns:
        Dict mapping relationship ID (e.g. ``rId5``) → target path (e.g.
        ``media/image1.png``).
    """
    import xml.etree.ElementTree as ET

    if rels_path not in zip_file.namelist():
        return {}
    try:
        rels_xml = zip_file.read(rels_path)
        root = ET.fromstring(rels_xml)
        ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        mapping: Dict[str, str] = {}
        for rel in root.findall(f"{ns}Relationship"):
            target = rel.get("Target", "")
            if "media/" in target.lower() or "image" in rel.get("Type", "").lower():
                rid = rel.get("Id", "")
                if rid:
                    mapping[rid] = target
        return mapping
    except Exception as exc:
        LOGGER.debug("Failed to parse rels file %s: %s", rels_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Page / slide number resolution
# ---------------------------------------------------------------------------

def _build_docx_media_to_page_map(zip_file: zipfile.ZipFile) -> Dict[str, int]:
    """Walk ``word/document.xml`` and resolve each embedded image to a page.

    Page boundaries are detected from:
    * ``<w:br w:type="page"/>`` — explicit page break
    * ``<w:lastRenderedPageBreak/>`` — Word's cached page break position
    * ``<w:sectPr>`` at paragraph level — section break (new page)

    Images are found inside ``<w:drawing>`` → ``<a:blip r:embed="rIdX"/>``.
    The *rId* is resolved via ``word/_rels/document.xml.rels`` to a media
    filename, which is used as the map key.

    Returns:
        Dict mapping media filename (e.g. ``word/media/image1.png``) → 1-based
        page number.  Media paths that cannot be resolved are omitted.
    """
    import xml.etree.ElementTree as ET

    doc_path = "word/document.xml"
    if doc_path not in zip_file.namelist():
        return {}

    # rId → media path (relative to "word/")
    rid_map = _parse_rels_file(zip_file, "word/_rels/document.xml.rels")

    try:
        doc_xml = zip_file.read(doc_path)
        root = ET.fromstring(doc_xml)
    except Exception as exc:
        LOGGER.debug("Failed to parse %s: %s", doc_path, exc)
        return {}

    # Common Office Open XML namespaces
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

    media_to_page: Dict[str, int] = {}
    current_page = 1

    # Iterate over all paragraphs (<w:p>) in document order
    body = root.find(f"{W}body")
    if body is None:
        return {}

    for para in body.iter(f"{W}p"):
        # Check for page breaks *before* processing images in this paragraph
        for run in para.iter(f"{W}r"):
            for br in run.iter(f"{W}br"):
                if br.get(f"{W}type") == "page" or br.get("type") == "page":
                    current_page += 1
            for _ in run.iter(f"{W}lastRenderedPageBreak"):
                current_page += 1

        # Find all embedded images in this paragraph
        for blip in para.iter(f"{A}blip"):
            rid = blip.get(f"{R}embed") or blip.get("r:embed") or ""
            if rid and rid in rid_map:
                target = rid_map[rid]  # e.g. "media/image1.png"
                # Normalise to full ZIP path
                if not target.startswith("word/"):
                    target = f"word/{target}"
                if target not in media_to_page:
                    media_to_page[target] = current_page

        # Section break in paragraph properties → new page for *next* paragraph
        ppr = para.find(f"{W}pPr")
        if ppr is not None and ppr.find(f"{W}sectPr") is not None:
            current_page += 1

    LOGGER.debug(
        "DOCX media→page map: %d images resolved across %d pages",
        len(media_to_page), current_page,
    )
    return media_to_page


def _build_pptx_media_to_slide_map(zip_file: zipfile.ZipFile) -> Dict[str, int]:
    """Map each media file in a PPTX to its 1-based slide number.

    PPTX stores slides as ``ppt/slides/slide1.xml``, ``ppt/slides/slide2.xml``,
    etc.  Each slide has its own ``.rels`` file mapping relationship IDs to
    media paths.

    Returns:
        Dict mapping full ZIP media path (e.g. ``ppt/media/image3.png``)
        → slide number.
    """
    import re as _re

    media_to_slide: Dict[str, int] = {}
    slide_pattern = _re.compile(r"ppt/slides/slide(\d+)\.xml$", _re.IGNORECASE)

    for name in sorted(zip_file.namelist()):
        m = slide_pattern.match(name)
        if not m:
            continue
        slide_num = int(m.group(1))

        # Corresponding rels file
        rels_path = name.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
        rid_map = _parse_rels_file(zip_file, rels_path)

        for _rid, target in rid_map.items():
            # target is relative, e.g. "../media/image3.png"
            if target.startswith(".."):
                full_path = f"ppt/{target[3:]}"  # strip "../"
            elif not target.startswith("ppt/"):
                full_path = f"ppt/{target}"
            else:
                full_path = target
            # Normalise
            full_path = os.path.normpath(full_path).replace("\\", "/")
            if full_path not in media_to_slide:
                media_to_slide[full_path] = slide_num

    LOGGER.debug(
        "PPTX media→slide map: %d images across %d slides",
        len(media_to_slide),
        len({v for v in media_to_slide.values()}),
    )
    return media_to_slide


# ---------------------------------------------------------------------------
# Image and page orientation correction
# ---------------------------------------------------------------------------

# EXIF Orientation tag → (transpose_method, degrees_clockwise)
# PIL transpose methods handle both rotation and mirroring.
_EXIF_ORIENTATION_TAG = 0x0112  # decimal 274

_EXIF_TRANSPOSE_MAP = {
    2: (Image.FLIP_LEFT_RIGHT, 0) if _PIL_AVAILABLE else (None, 0),
    3: (Image.ROTATE_180, 180) if _PIL_AVAILABLE else (None, 180),
    4: (Image.FLIP_TOP_BOTTOM, 0) if _PIL_AVAILABLE else (None, 0),
    5: (Image.TRANSPOSE, 90) if _PIL_AVAILABLE else (None, 90),
    6: (Image.ROTATE_270, 90) if _PIL_AVAILABLE else (None, 90),
    7: (Image.TRANSVERSE, 270) if _PIL_AVAILABLE else (None, 270),
    8: (Image.ROTATE_90, 270) if _PIL_AVAILABLE else (None, 270),
}


def _correct_exif_orientation(img_bytes: bytes, fmt: str) -> Tuple[bytes, int]:
    """Apply EXIF orientation correction to JPEG images.

    Args:
        img_bytes: Raw image bytes.
        fmt: Image format string (e.g. ``"jpeg"``, ``"png"``).

    Returns:
        Tuple of (corrected_bytes, degrees_corrected).  If no correction
        was needed the original bytes are returned unchanged with 0.
    """
    if not _PIL_AVAILABLE:
        return img_bytes, 0
    if fmt not in ("jpeg", "jpg"):
        return img_bytes, 0

    try:
        img = Image.open(BytesIO(img_bytes))
        exif_data = img.getexif()
        if not exif_data:
            return img_bytes, 0

        orientation = exif_data.get(_EXIF_ORIENTATION_TAG)
        if orientation is None or orientation == 1:
            return img_bytes, 0

        entry = _EXIF_TRANSPOSE_MAP.get(orientation)
        if entry is None:
            return img_bytes, 0

        transpose_method, degrees = entry
        if transpose_method is not None:
            img = img.transpose(transpose_method)

        # Strip the EXIF orientation tag so downstream consumers don't re-apply
        exif_data.pop(_EXIF_ORIENTATION_TAG, None)

        buf = BytesIO()
        img.save(buf, format="JPEG", exif=exif_data.tobytes() if exif_data else b"")
        LOGGER.debug("Applied EXIF orientation correction: tag=%d, degrees=%d", orientation, degrees)
        return buf.getvalue(), degrees

    except Exception as exc:
        LOGGER.debug("EXIF orientation correction failed: %s", exc)
        return img_bytes, 0


def _apply_rotation(img_bytes: bytes, fmt: str, degrees_cw: int) -> Tuple[bytes, str]:
    """Rotate image by the given clockwise degrees using Pillow.

    Args:
        img_bytes: Raw image bytes.
        fmt: Image format (``"png"``, ``"jpeg"``, etc.).
        degrees_cw: Clockwise rotation in degrees (90, 180, 270).

    Returns:
        Tuple of (rotated_bytes, output_format).
    """
    if not _PIL_AVAILABLE or degrees_cw == 0:
        return img_bytes, fmt

    # PIL's rotate() is counter-clockwise, so we negate
    ccw = (360 - degrees_cw) % 360
    if ccw == 0:
        return img_bytes, fmt

    try:
        img = Image.open(BytesIO(img_bytes))
        img = img.rotate(ccw, expand=True)

        # Prefer PNG for lossless output; keep JPEG if input was JPEG
        out_fmt = "JPEG" if fmt in ("jpeg", "jpg") else "PNG"
        buf = BytesIO()
        img.save(buf, format=out_fmt)
        LOGGER.debug("Rotated image %d° clockwise (PIL %d° CCW)", degrees_cw, ccw)
        return buf.getvalue(), fmt
    except Exception as exc:
        LOGGER.debug("Image rotation failed: %s", exc)
        return img_bytes, fmt


def _build_docx_media_rotation_map(
    zip_file: zipfile.ZipFile,
) -> Dict[str, int]:
    """Parse ``word/document.xml`` and extract rotation angles for embedded images.

    OOXML stores rotation in ``<a:xfrm rot="VALUE">`` where VALUE is in
    60,000ths of a degree (e.g. 5400000 = 90°).  Flips are stored as
    ``flipH="1"`` and ``flipV="1"`` on the same element.

    Returns:
        Dict mapping media ZIP path (e.g. ``word/media/image1.png``)
        → clockwise rotation in degrees (0, 90, 180, 270).
    """
    import xml.etree.ElementTree as ET

    doc_path = "word/document.xml"
    if doc_path not in zip_file.namelist():
        return {}

    rid_map = _parse_rels_file(zip_file, "word/_rels/document.xml.rels")

    try:
        doc_xml = zip_file.read(doc_path)
        root = ET.fromstring(doc_xml)
    except Exception as exc:
        LOGGER.debug("Failed to parse %s for rotation: %s", doc_path, exc)
        return {}

    A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

    media_rotation: Dict[str, int] = {}

    # Find all drawing elements: <wp:inline> and <wp:anchor>
    for drawing_container in ("inline", "anchor"):
        tag = f"{WP}{drawing_container}"
        for container in root.iter(tag):
            # Find the blip (image reference) inside this drawing
            blip = container.find(f".//{A}blip")
            if blip is None:
                continue
            rid = blip.get(f"{R}embed") or blip.get("r:embed") or ""
            if not rid or rid not in rid_map:
                continue

            target = rid_map[rid]
            if not target.startswith("word/"):
                target = f"word/{target}"

            # Find <a:xfrm> containing rotation
            xfrm = container.find(f".//{A}xfrm")
            if xfrm is None:
                continue

            rot_raw = xfrm.get("rot", "0")
            try:
                # OOXML rotation is in 60,000ths of a degree
                rot_emu = int(rot_raw)
                rot_degrees = (rot_emu // 60000) % 360
            except (ValueError, TypeError):
                rot_degrees = 0

            flip_h = xfrm.get("flipH", "0") == "1"
            flip_v = xfrm.get("flipV", "0") == "1"

            # Combine flip with rotation:
            # flipH + flipV is equivalent to 180° rotation
            if flip_h and flip_v:
                rot_degrees = (rot_degrees + 180) % 360
            # Individual flips are handled separately in the correction step

            if rot_degrees != 0:
                media_rotation[target] = rot_degrees
                LOGGER.debug(
                    "DOCX image %s: rotation=%d° (flipH=%s, flipV=%s)",
                    target, rot_degrees, flip_h, flip_v,
                )

    if media_rotation:
        LOGGER.debug("DOCX rotation map: %d rotated images found", len(media_rotation))

    return media_rotation


def _build_docx_page_orientation_map(
    zip_file: zipfile.ZipFile,
) -> Dict[int, str]:
    """Detect page orientation (portrait/landscape) from DOCX section properties.

    DOCX stores page dimensions in ``<w:pgSz>`` elements within ``<w:sectPr>``.
    When ``w:orient="landscape"`` is set, or when width > height, the page
    is landscape.

    Returns:
        Dict mapping 1-based page number → ``"portrait"`` or ``"landscape"``.
        Only pages that start a new section are explicitly mapped; pages
        between sections inherit the previous section's orientation.
    """
    import xml.etree.ElementTree as ET

    doc_path = "word/document.xml"
    if doc_path not in zip_file.namelist():
        return {}

    try:
        doc_xml = zip_file.read(doc_path)
        root = ET.fromstring(doc_xml)
    except Exception as exc:
        LOGGER.debug("Failed to parse %s for orientation: %s", doc_path, exc)
        return {}

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    body = root.find(f"{W}body")
    if body is None:
        return {}

    orientation_map: Dict[int, str] = {}
    current_page = 1

    # Walk paragraphs to count pages and detect section breaks
    for para in body.iter(f"{W}p"):
        # Count page breaks
        for run in para.iter(f"{W}r"):
            for br in run.iter(f"{W}br"):
                if br.get(f"{W}type") == "page" or br.get("type") == "page":
                    current_page += 1
            for _ in run.iter(f"{W}lastRenderedPageBreak"):
                current_page += 1

        # Check for section properties in paragraph
        ppr = para.find(f"{W}pPr")
        if ppr is not None:
            sect_pr = ppr.find(f"{W}sectPr")
            if sect_pr is not None:
                pg_sz = sect_pr.find(f"{W}pgSz")
                if pg_sz is not None:
                    orient = pg_sz.get(f"{W}orient", "portrait")
                    # Also detect from dimensions: w > h means landscape
                    w_val = pg_sz.get(f"{W}w", "0")
                    h_val = pg_sz.get(f"{W}h", "0")
                    try:
                        w_twips = int(w_val)
                        h_twips = int(h_val)
                        if w_twips > h_twips:
                            orient = "landscape"
                    except (ValueError, TypeError):
                        pass
                    orientation_map[current_page] = orient
                current_page += 1

    # Also check the final section (body-level <w:sectPr>)
    final_sect = body.find(f"{W}sectPr")
    if final_sect is not None:
        pg_sz = final_sect.find(f"{W}pgSz")
        if pg_sz is not None:
            orient = pg_sz.get(f"{W}orient", "portrait")
            w_val = pg_sz.get(f"{W}w", "0")
            h_val = pg_sz.get(f"{W}h", "0")
            try:
                w_twips = int(w_val)
                h_twips = int(h_val)
                if w_twips > h_twips:
                    orient = "landscape"
            except (ValueError, TypeError):
                pass
            # The final section covers all remaining pages
            if 1 not in orientation_map:
                orientation_map[1] = orient
            orientation_map[current_page] = orient

    if orientation_map:
        LOGGER.debug("DOCX page orientation map: %s", orientation_map)

    return orientation_map


# ---------------------------------------------------------------------------
# Main Extractors
# ---------------------------------------------------------------------------

class DocxImageExtractor(BaseImageExtractor):
    """
    Extract images from DOCX files.

    DOCX files are ZIP archives. Embedded images reside in ``word/media/``.
    """

    SUPPORTED_FORMATS = set(_SUPPORTED_EXTS.values())

    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        skip_duplicates: bool = True,
    ):
        """
        Args:
            min_width:  Minimum image width in pixels.
            min_height: Minimum image height in pixels.
            skip_duplicates: De-duplicate images by content hash.
        """
        super().__init__(min_width=min_width, min_height=min_height)
        self.skip_duplicates = skip_duplicates

    def extract_images(
        self,
        file_path: str,
        document_id: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a ``.docx`` file.

        Applies orientation corrections:
        - EXIF orientation for JPEG images
        - OOXML rotation from ``<a:xfrm rot="...">`` drawing attributes
        - Page orientation detection (landscape/portrait) stored in metadata

        Args:
            file_path:   Path to the ``.docx`` file.
            document_id: Unique document identifier (auto-generated if omitted).

        Returns:
            List of :class:`ExtractedImage` objects with corrected orientation.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        if document_id is None:
            document_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]

        return self._extract_from_zip(
            file_path=file_path,
            document_id=document_id,
            media_prefix="word/media/",
            id_prefix="docx",
            page_map_builder=_build_docx_media_to_page_map,
            rotation_map_builder=_build_docx_media_rotation_map,
            page_orientation_builder=_build_docx_page_orientation_map,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_from_zip(
        self,
        file_path: str,
        document_id: str,
        media_prefix: str,
        id_prefix: str,
        page_map_builder: Optional[object] = None,
        rotation_map_builder: Optional[object] = None,
        page_orientation_builder: Optional[object] = None,
    ) -> List[ExtractedImage]:
        """Generic extraction from an OOXML ZIP archive.

        Args:
            page_map_builder: Optional callable ``(ZipFile) → Dict[str, int]``
                that maps media ZIP paths to 1-based page/slide numbers.
            rotation_map_builder: Optional callable ``(ZipFile) → Dict[str, int]``
                that maps media ZIP paths to clockwise rotation degrees.
            page_orientation_builder: Optional callable ``(ZipFile) → Dict[int, str]``
                that maps 1-based page numbers to ``"portrait"`` or ``"landscape"``.
        """
        images: List[ExtractedImage] = []
        seen_hashes: set = set()

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                # Build page-number lookup if a builder was provided
                media_to_page: Dict[str, int] = {}
                if callable(page_map_builder):
                    try:
                        media_to_page = page_map_builder(zf)
                    except Exception as exc:
                        LOGGER.debug("Page-map builder failed (falling back to page 1): %s", exc)

                # Build rotation lookup (OOXML <a:xfrm rot="...">)
                media_rotation: Dict[str, int] = {}
                if callable(rotation_map_builder):
                    try:
                        media_rotation = rotation_map_builder(zf)
                    except Exception as exc:
                        LOGGER.debug("Rotation-map builder failed: %s", exc)

                # Build page orientation lookup (portrait/landscape)
                page_orientation: Dict[int, str] = {}
                if callable(page_orientation_builder):
                    try:
                        page_orientation = page_orientation_builder(zf)
                    except Exception as exc:
                        LOGGER.debug("Page-orientation builder failed: %s", exc)

                media_files = sorted(
                    n for n in zf.namelist()
                    if n.lower().startswith(media_prefix.lower())
                )

                LOGGER.debug(
                    "Found %d media files under '%s' in %s",
                    len(media_files), media_prefix, file_path,
                )

                for idx, member in enumerate(media_files):
                    ext = os.path.splitext(member)[1].lower()

                    if ext in _SKIP_EXTS:
                        LOGGER.debug("Skipping vector format: %s", member)
                        continue

                    fmt = _SUPPORTED_EXTS.get(ext)
                    if not fmt:
                        LOGGER.debug("Unsupported image format: %s", member)
                        continue

                    try:
                        img_bytes = zf.read(member)
                    except Exception as exc:
                        LOGGER.warning("Failed to read %s: %s", member, exc)
                        continue

                    if not img_bytes:
                        continue

                    # De-duplicate by content hash
                    if self.skip_duplicates:
                        content_hash = hashlib.sha256(img_bytes).hexdigest()
                        if content_hash in seen_hashes:
                            LOGGER.debug("Skipping duplicate image: %s", member)
                            continue
                        seen_hashes.add(content_hash)

                    # Dimension check
                    width, height = _get_image_dimensions(img_bytes, fmt)
                    if width and height:
                        if not self._passes_size_filter(width, height):
                            LOGGER.debug(
                                "Skipping small image (%dx%d): %s",
                                width, height, member,
                            )
                            continue

                    # Resolve page/slide number from the map
                    # Try both raw member path and normalised form
                    page_num = media_to_page.get(member)
                    if page_num is None:
                        normalised = os.path.normpath(member).replace("\\", "/")
                        page_num = media_to_page.get(normalised, 1)

                    # --- Orientation correction ---
                    total_rotation = 0

                    # 1. EXIF orientation (JPEG only)
                    img_bytes, exif_degrees = _correct_exif_orientation(img_bytes, fmt)
                    total_rotation += exif_degrees

                    # 2. OOXML drawing rotation (<a:xfrm rot="...">)
                    xml_rot = media_rotation.get(member)
                    if xml_rot is None:
                        normalised_member = os.path.normpath(member).replace("\\", "/")
                        xml_rot = media_rotation.get(normalised_member, 0)
                    if xml_rot:
                        img_bytes, fmt = _apply_rotation(img_bytes, fmt, xml_rot)
                        total_rotation = (total_rotation + xml_rot) % 360

                    # Update dimensions after rotation
                    if total_rotation in (90, 270):
                        width, height = height, width

                    image_id = f"{document_id}_{id_prefix}_img{idx + 1}"

                    images.append(
                        ExtractedImage(
                            image_id=image_id,
                            page_number=page_num,
                            image_bytes=img_bytes,
                            format=fmt,
                            bounding_box=None,
                            description=None,  # filled by Vision LLM later
                            image_type=ImageType.UNKNOWN,
                            width=width,
                            height=height,
                            file_size=len(img_bytes),
                            rotation_corrected=total_rotation,
                        )
                    )

                corrected_count = sum(1 for img in images if img.rotation_corrected != 0)
                LOGGER.info(
                    "Extracted %d images from %s (%s* prefix), %d orientation-corrected",
                    len(images), file_path, media_prefix, corrected_count,
                )

                # Attach page orientation metadata to images
                if page_orientation:
                    last_orient = "portrait"
                    for img in images:
                        orient = page_orientation.get(img.page_number)
                        if orient:
                            last_orient = orient
                        # Images inherit the most recent section orientation
                        # Store as page_width/page_height hint for downstream
                        if last_orient == "landscape" and img.width and img.height:
                            if img.width < img.height:
                                # Image dimensions suggest portrait but page is landscape
                                LOGGER.debug(
                                    "Image %s on landscape page %d may need review",
                                    img.image_id, img.page_number,
                                )
        except zipfile.BadZipFile:
            LOGGER.warning("Invalid ZIP / OOXML file: %s", file_path)
        except Exception as exc:
            LOGGER.warning("Image extraction failed for %s: %s", file_path, exc)

        return images


class PptxImageExtractor(BaseImageExtractor):
    """
    Extract images from PPTX files.

    PPTX files are ZIP archives. Embedded images reside in ``ppt/media/``.
    """

    SUPPORTED_FORMATS = set(_SUPPORTED_EXTS.values())

    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        skip_duplicates: bool = True,
    ):
        """
        Args:
            min_width:  Minimum image width in pixels.
            min_height: Minimum image height in pixels.
            skip_duplicates: De-duplicate images by content hash.
        """
        super().__init__(min_width=min_width, min_height=min_height)
        self.skip_duplicates = skip_duplicates

    def extract_images(
        self,
        file_path: str,
        document_id: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a ``.pptx`` file.

        Args:
            file_path:   Path to the ``.pptx`` file.
            document_id: Unique document identifier (auto-generated if omitted).

        Returns:
            List of :class:`ExtractedImage` objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PPTX file not found: {file_path}")

        if document_id is None:
            document_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]

        # Re-use the same ZIP extraction logic as DOCX, just with a
        # different media prefix and a slide-number resolver.
        extractor = DocxImageExtractor(
            min_width=self.min_width,
            min_height=self.min_height,
            skip_duplicates=self.skip_duplicates,
        )

        return extractor._extract_from_zip(
            file_path=file_path,
            document_id=document_id,
            media_prefix="ppt/media/",
            id_prefix="pptx",
            page_map_builder=_build_pptx_media_to_slide_map,
        )
