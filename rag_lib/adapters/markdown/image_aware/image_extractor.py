"""
Image extraction utilities for PDF documents.
"""

import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from .models import ExtractedImage, BoundingBox, ImageType
from .base_extractor import BaseImageExtractor

LOGGER = logging.getLogger(__name__)

try:
    import pypdfium2 as pdfium  # Apache-2.0 / BSD-3
    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False

# pypdfium2's underlying C library (PDFium) is NOT thread-safe.
# Docling also uses pypdfium2 internally with its own lock.  We must
# share the same lock so that image extraction never runs concurrently
# with Docling's threaded pipeline stages.
try:
    from docling.utils.locks import pypdfium2_lock as _pdfium_lock
except ImportError:
    # Docling not installed — create a standalone lock.
    _pdfium_lock = threading.Lock()


class ImageExtractor(BaseImageExtractor):
    """
    Extract images from PDF documents using pypdfium2.
    
    Supports:
    - Embedded images in PDFs
    - Vector-drawn graphics (charts, diagrams, infographics)
    - Full page renders for complex layouts
    - Image classification (chart, table, diagram, etc.)
    """

    # Minimum PATH/FORM objects on a page to consider it a vector graphic
    _MIN_VECTOR_OBJECTS = 15
    # Vertical gap (as fraction of page height) that splits clusters
    _VECTOR_GAP_THRESHOLD = 0.03
    # Maximum height of a cluster (fraction of tallest cluster) to treat as legend
    _LEGEND_HEIGHT_RATIO = 0.15

    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.85,
        extract_full_pages: bool = False,
        page_render_dpi: int = 150,
    ):
        """
        Initialize the image extractor.
        
        Args:
            min_width: Minimum image width to extract (pixels)
            min_height: Minimum image height to extract (pixels)
            min_area_ratio: Minimum ratio of image area to page area
            max_area_ratio: Maximum ratio of image area to page area. Images
                covering more of the page than this (e.g. full-page scanned
                backgrounds or decorative banners) are skipped, since their
                content is already captured as text by the base markdown
                adapter (Docling/Vision LLM) and re-embedding + describing
                them as images is redundant. Set to 1.0 to disable.
            extract_full_pages: Also extract full page renders
            page_render_dpi: DPI for full page renders
        """
        if not PYPDFIUM2_AVAILABLE:
            raise ImportError(
                "pypdfium2 is required for image extraction. "
                "Install with: pip install pypdfium2"
            )
        
        super().__init__(min_width=min_width, min_height=min_height)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.extract_full_pages = extract_full_pages
        self.page_render_dpi = page_render_dpi
    
    def extract_images(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        pages: Optional[List[int]] = None,
        vector_pages: Optional[List[int]] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            document_id: Unique identifier for the document
            pages: Specific pages to extract from (1-indexed), None for all
            vector_pages: Deprecated / ignored. Vector-graphics fallback is
                now attempted on every page that has no raster images.
                The ``_MIN_VECTOR_OBJECTS`` threshold guards against
                false positives.
            
        Returns:
            List of ExtractedImage objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(file_path)
        
        # Acquire the process-wide pypdfium2 lock to prevent concurrent
        # access with Docling's threaded pipeline stages.  PDFium's C
        # library is not thread-safe — concurrent calls cause silent
        # failures (0 images returned) or segfaults.
        with _pdfium_lock:
            doc = pdfium.PdfDocument(file_path)
            images = []
            
            try:
                total_pages = len(doc)
                target_pages = pages if pages else list(range(1, total_pages + 1))

                for page_num in target_pages:
                    if page_num < 1 or page_num > total_pages:
                        continue
                    
                    page = doc[page_num - 1]  # 0-indexed
                    page_images = self._extract_page_images(
                        doc, page, page_num, document_id,
                        allow_vector=True,
                    )
                    images.extend(page_images)
                    
                    # Optionally extract full page render
                    if self.extract_full_pages:
                        full_page_img = self._render_full_page(
                            page, page_num, document_id
                        )
                        if full_page_img:
                            images.append(full_page_img)
                    
                    LOGGER.debug(
                        "Extracted %d images from page %d",
                        len(page_images), page_num
                    )
            finally:
                doc.close()
        
        LOGGER.info(
            "Extracted %d total images from %s",
            len(images), file_path
        )
        return images
    
    def _extract_page_images(
        self,
        doc: "pdfium.PdfDocument",
        page: "pdfium.PdfPage",
        page_num: int,
        document_id: str,
        allow_vector: bool = False,
    ) -> List[ExtractedImage]:
        """Extract embedded images from a single page."""
        images = []
        page_width, page_height = page.get_size()
        page_area = page_width * page_height
        
        # Iterate over image objects only (filter at C level)
        for img_index, obj in enumerate(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])):
            try:
                
                # Get image metadata
                try:
                    pil_image = obj.get_bitmap().to_pil()
                except Exception:
                    continue
                
                width, height = pil_image.size
                
                # Check minimum size
                if not self._passes_size_filter(width, height):
                    continue
                
                # Get bounding box from object position
                pos = obj.get_pos()
                bbox = BoundingBox(
                    x1=pos[0], y1=pos[1],
                    x2=pos[2], y2=pos[3],
                )
                
                # Check minimum area ratio
                if bbox.area / page_area < self.min_area_ratio:
                    continue

                # Check maximum area ratio — skip near-full-page images
                # (scanned backgrounds/banners) whose content duplicates
                # what the base text adapter already extracted.
                if bbox.area / page_area > self.max_area_ratio:
                    LOGGER.debug(
                        "Skipping near-full-page image on page %d (area ratio %.2f > %.2f)",
                        page_num, bbox.area / page_area, self.max_area_ratio,
                    )
                    continue
                
                # Convert to PNG bytes
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG", optimize=True)
                image_bytes = buf.getvalue()
                
                # Generate image ID
                image_id = f"{document_id}_p{page_num}_img{img_index + 1}"
                
                # Classify image type
                image_type = self._classify_image(width, height, bbox, page_area)
                
                extracted = ExtractedImage(
                    image_id=image_id,
                    page_number=page_num,
                    image_bytes=image_bytes,
                    format="png",
                    bounding_box=bbox,
                    image_type=image_type,
                    width=width,
                    height=height,
                    file_size=len(image_bytes),
                    page_width=page_width,
                    page_height=page_height,
                )
                images.append(extracted)
                
            except Exception as e:
                LOGGER.warning(
                    "Failed to extract image %d from page %d: %s",
                    img_index, page_num, e
                )

        # Vector-graphics fallback: when no embedded raster images were found
        # AND the page is in the allowed vector pages list, check for
        # significant PATH/FORM objects (charts, diagrams drawn as vector
        # paths) and render cropped screenshots of those regions.
        if not images and allow_vector:
            vector_images = self._extract_vector_graphics(
                page, page_num, document_id, page_width, page_height,
            )
            images.extend(vector_images)

        return images
    
    def _render_full_page(
        self,
        page: "pdfium.PdfPage",
        page_num: int,
        document_id: str,
    ) -> Optional[ExtractedImage]:
        """Render full page as an image."""
        try:
            scale = self.page_render_dpi / 72
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG", optimize=True)
            image_bytes = buf.getvalue()
            width, height = pil_image.size
            
            image_id = f"{document_id}_p{page_num}_full"
            
            return ExtractedImage(
                image_id=image_id,
                page_number=page_num,
                image_bytes=image_bytes,
                format="png",
                bounding_box=BoundingBox(0, 0, width, height),
                image_type=ImageType.UNKNOWN,
                width=width,
                height=height,
                file_size=len(image_bytes),
            )
        except Exception as e:
            LOGGER.warning("Failed to render full page %d: %s", page_num, e)
            return None
    
    def _classify_image(
        self,
        width: int,
        height: int,
        bbox: Optional[BoundingBox],
        page_area: float,
    ) -> ImageType:
        """
        Classify the type of image based on dimensions and position.
        
        This is a heuristic classification. For more accurate classification,
        use Vision LLM description.
        """
        aspect_ratio = width / height if height > 0 else 1
        
        # Small images are likely logos
        if width < 200 and height < 200:
            return ImageType.LOGO
        
        # Very wide images might be charts/graphs
        if aspect_ratio > 2.0:
            return ImageType.CHART
        
        # Square-ish large images might be infographics
        if bbox:
            coverage = bbox.area / page_area
            if coverage > 0.3 and 0.5 < aspect_ratio < 2.0:
                return ImageType.INFOGRAPHIC
        
        return ImageType.UNKNOWN
    
    # ------------------------------------------------------------------
    #  Vector-graphics detection & extraction
    # ------------------------------------------------------------------

    def _extract_vector_graphics(
        self,
        page: "pdfium.PdfPage",
        page_num: int,
        document_id: str,
        page_width: float,
        page_height: float,
    ) -> List[ExtractedImage]:
        """Detect and extract vector-drawn visuals from a page.

        Called as a fallback when ``_extract_page_images`` finds zero
        embedded raster images.  Collects PATH and FORM object bounding
        boxes, clusters them by vertical gap, merges back small legend
        clusters, and renders a cropped screenshot for each region.
        """
        bboxes = self._collect_vector_bboxes(page)

        if len(bboxes) < self._MIN_VECTOR_OBJECTS:
            return []

        # Check combined vector area covers a meaningful page portion
        page_area = page_width * page_height
        all_left = min(b[0] for b in bboxes)
        all_bottom = min(b[1] for b in bboxes)
        all_right = max(b[2] for b in bboxes)
        all_top = max(b[3] for b in bboxes)
        combined_area = (all_right - all_left) * (all_top - all_bottom)

        if combined_area / page_area < self.min_area_ratio:
            return []

        # Cluster vertically and merge small legend clusters back
        clusters = self._cluster_by_vertical_gap(
            bboxes, page_height, self._VECTOR_GAP_THRESHOLD,
        )
        clusters = self._merge_legend_clusters(clusters, self._LEGEND_HEIGHT_RATIO)

        # Render a cropped screenshot per cluster
        scale = self.page_render_dpi / 72
        images: List[ExtractedImage] = []

        for idx, cluster_bboxes in enumerate(clusters):
            c_left = min(b[0] for b in cluster_bboxes)
            c_bottom = min(b[1] for b in cluster_bboxes)
            c_right = max(b[2] for b in cluster_bboxes)
            c_top = max(b[3] for b in cluster_bboxes)

            # Skip text-like clusters: wide aspect ratio + low density
            # indicates paragraph text rendered as vector outlines, not charts.
            c_w = c_right - c_left
            c_h = c_top - c_bottom
            if c_h > 0 and c_w / c_h > 2.0:
                obj_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in cluster_bboxes)
                cluster_raw_area = c_w * c_h
                density = obj_area / cluster_raw_area if cluster_raw_area > 0 else 0
                if density < 0.5:
                    continue

            # 5 % padding so edges aren't clipped
            pad_x = (c_right - c_left) * 0.05
            pad_y = (c_top - c_bottom) * 0.05
            c_left = max(0, c_left - pad_x)
            c_bottom = max(0, c_bottom - pad_y)
            c_right = min(page_width, c_right + pad_x)
            c_top = min(page_height, c_top + pad_y)

            cluster_area = (c_right - c_left) * (c_top - c_bottom)
            if cluster_area / page_area < self.min_area_ratio:
                continue

            # Skip near-full-page clusters (e.g. a full-page background
            # rendered as vector paths) — same rationale as the raster
            # image max_area_ratio filter above: this content duplicates
            # what the base text adapter already extracted.
            if cluster_area / page_area > self.max_area_ratio:
                LOGGER.debug(
                    "Skipping near-full-page vector region on page %d (area ratio %.2f > %.2f)",
                    page_num, cluster_area / page_area, self.max_area_ratio,
                )
                continue

            extracted = self._render_vector_region(
                page, page_num, document_id, idx,
                c_left, c_bottom, c_right, c_top,
                page_width, page_height, scale,
            )
            if extracted:
                images.append(extracted)

        if images:
            LOGGER.info(
                "Detected %d vector-graphic region(s) on page %d "
                "(%d PATH/FORM objects)",
                len(images), page_num, len(bboxes),
            )

        return images

    @staticmethod
    def _collect_vector_bboxes(
        page: "pdfium.PdfPage",
    ) -> List[Tuple[float, float, float, float]]:
        """Return bounding boxes of significant PATH and FORM objects."""
        bboxes: List[Tuple[float, float, float, float]] = []
        for obj in page.get_objects(
            filter=[pdfium.raw.FPDF_PAGEOBJ_PATH, pdfium.raw.FPDF_PAGEOBJ_FORM],
        ):
            try:
                left, bottom, right, top = obj.get_pos()
                # Skip hairlines, borders, and other tiny decorative elements
                if (right - left) < 5 and (top - bottom) < 5:
                    continue
                bboxes.append((left, bottom, right, top))
            except Exception:
                continue
        return bboxes

    @staticmethod
    def _cluster_by_vertical_gap(
        bboxes: List[Tuple[float, float, float, float]],
        page_height: float,
        gap_threshold_ratio: float,
    ) -> List[List[Tuple[float, float, float, float]]]:
        """Split bounding boxes into clusters separated by vertical gaps.

        Sorts objects by their vertical midpoint, then cuts at any gap
        larger than *gap_threshold_ratio* × *page_height*.
        """
        if not bboxes:
            return []

        gap_threshold = gap_threshold_ratio * page_height

        # Sort by vertical midpoint (ascending – bottom of page first)
        sorted_bboxes = sorted(bboxes, key=lambda b: (b[1] + b[3]) / 2)

        clusters: List[List[Tuple[float, float, float, float]]] = [[sorted_bboxes[0]]]
        for bbox in sorted_bboxes[1:]:
            prev_top = max(b[3] for b in clusters[-1])
            cur_bottom = bbox[1]
            if cur_bottom - prev_top > gap_threshold:
                clusters.append([bbox])
            else:
                clusters[-1].append(bbox)

        return clusters

    @staticmethod
    def _merge_legend_clusters(
        clusters: List[List[Tuple[float, float, float, float]]],
        legend_height_ratio: float,
    ) -> List[List[Tuple[float, float, float, float]]]:
        """Merge small legend-like clusters into their nearest neighbour.

        A cluster whose height is less than *legend_height_ratio* of the
        tallest cluster is considered a legend and merged into the
        adjacent cluster that is closest vertically.
        """
        if len(clusters) <= 1:
            return clusters

        def _cluster_height(c):
            return max(b[3] for b in c) - min(b[1] for b in c)

        max_height = max(_cluster_height(c) for c in clusters)
        if max_height == 0:
            return clusters

        threshold = legend_height_ratio * max_height

        merged: List[List[Tuple[float, float, float, float]]] = []
        pending_legend = None

        for cluster in clusters:
            if _cluster_height(cluster) < threshold:
                # Buffer this small cluster to merge later
                if pending_legend is None:
                    pending_legend = cluster
                else:
                    pending_legend.extend(cluster)
            else:
                # Flush any pending legend into this (large) cluster
                if pending_legend is not None:
                    cluster = pending_legend + cluster
                    pending_legend = None
                merged.append(cluster)

        # If a trailing legend remains, attach to the last large cluster
        if pending_legend is not None:
            if merged:
                merged[-1].extend(pending_legend)
            else:
                merged.append(pending_legend)

        return merged

    def _render_vector_region(
        self,
        page: "pdfium.PdfPage",
        page_num: int,
        document_id: str,
        region_index: int,
        left: float,
        bottom: float,
        right: float,
        top: float,
        page_width: float,
        page_height: float,
        scale: float,
    ) -> Optional[ExtractedImage]:
        """Render the full page and crop to the given PDF-coordinate region."""
        try:
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            # PDF coords → pixel coords  (PDF origin is bottom-left;
            # PIL origin is top-left, so Y is flipped).
            px_left = int(left * scale)
            px_upper = int((page_height - top) * scale)
            px_right = int(right * scale)
            px_lower = int((page_height - bottom) * scale)

            # Clamp to image bounds
            img_w, img_h = pil_image.size
            px_left = max(0, px_left)
            px_upper = max(0, px_upper)
            px_right = min(img_w, px_right)
            px_lower = min(img_h, px_lower)

            cropped = pil_image.crop((px_left, px_upper, px_right, px_lower))
            width, height = cropped.size

            if not self._passes_size_filter(width, height):
                return None

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            image_id = f"{document_id}_p{page_num}_img{region_index + 1}"

            return ExtractedImage(
                image_id=image_id,
                page_number=page_num,
                image_bytes=image_bytes,
                format="png",
                bounding_box=BoundingBox(left, bottom, right, top),
                image_type=ImageType.CHART,
                width=width,
                height=height,
                file_size=len(image_bytes),
                page_width=page_width,
                page_height=page_height,
            )
        except Exception as e:
            LOGGER.warning(
                "Failed to render vector region %d on page %d: %s",
                region_index, page_num, e,
            )
            return None

    # NOTE: _generate_document_id inherited from BaseImageExtractor
