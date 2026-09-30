"""Image chunk generation service.

Creates dedicated vector-search chunks for document images so that
visual content is independently retrievable.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)

# LLM Vision API limit for inline image bytes
_MAX_VLM_IMAGE_BYTES = 3_750_000


def _compress_for_vlm(png_bytes: bytes, page_num: int = 0) -> tuple[bytes, str]:
    """Shrink *png_bytes* to fit the LLM inline-image limit.

    Returns ``(image_bytes, format)`` where *format* is ``"png"`` or
    ``"jpeg"`` depending on whether compression was needed.
    """
    if len(png_bytes) <= _MAX_VLM_IMAGE_BYTES:
        return png_bytes, "png"

    try:
        from PIL import Image
    except ImportError:
        LOGGER.log_warning(
            ProcessLogger.STAGE_CHUNK,
            f"Pillow not installed — cannot compress page {page_num} image "
            f"({len(png_bytes)} bytes)",
        )
        return png_bytes, "png"

    img = Image.open(io.BytesIO(png_bytes))

    # Try optimised PNG first
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    if buf.tell() <= _MAX_VLM_IMAGE_BYTES:
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Page {page_num}: optimised PNG {len(png_bytes)} → {buf.tell()} bytes",
        )
        return buf.getvalue(), "png"

    # Convert to JPEG at decreasing quality
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    for quality in (90, 80, 70):
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= _MAX_VLM_IMAGE_BYTES:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                f"Page {page_num}: JPEG q={quality} reduced "
                f"{len(png_bytes)} → {buf.tell()} bytes",
            )
            return buf.getvalue(), "jpeg"

    # Downscale 75 % + JPEG
    new_w, new_h = int(rgb.width * 0.75), int(rgb.height * 0.75)
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    for quality in (85, 70):
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= _MAX_VLM_IMAGE_BYTES:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                f"Page {page_num}: resized JPEG q={quality} reduced "
                f"{len(png_bytes)} → {buf.tell()} bytes",
            )
            return buf.getvalue(), "jpeg"

    LOGGER.log_warning(
        ProcessLogger.STAGE_CHUNK,
        f"Page {page_num}: could not compress below "
        f"{_MAX_VLM_IMAGE_BYTES / 1_000_000:.1f} MB (best: {buf.tell()} bytes)",
    )
    return buf.getvalue(), "jpeg"


# ------------------------------------------------------------------
# Vision LLM prompts
# ------------------------------------------------------------------

_INFOGRAPHIC_DESCRIPTION_PROMPT = (
    "You are analysing a full page from a business document that was identified "
    "as an infographic or visually-rich page (charts, diagrams, tables with "
    "graphics, etc.).\n\n"
    "Describe the page comprehensively:\n"
    "1. What type of visual is it? (chart, infographic, dashboard, diagram, etc.)\n"
    "2. What data or information does it convey?\n"
    "3. List any specific numbers, percentages, or key metrics visible.\n"
    "4. Describe the layout and visual hierarchy.\n\n"
    "Be thorough — this description will be used for semantic search. "
    "Include all text visible in the image."
)

_RELEVANCE_TAGGING_PROMPT = (
    "You are evaluating whether an image extracted from a document is "
    "**content-relevant** or **decorative/unrelated**.\n\n"
    "Content-relevant images include: charts, graphs, diagrams, infographics, "
    "tables, screenshots, product photos, maps, org charts, process flows, "
    "and any image that conveys substantive information related to the "
    "document's topic.\n\n"
    "Decorative/unrelated images include: logos, icons, background patterns, "
    "page borders, watermarks, stock photos used purely for aesthetics, "
    "separator lines, and generic illustrations.\n\n"
    "Surrounding text context:\n```\n{context}\n```\n\n"
    "Respond with ONLY a JSON object (no markdown fences):\n"
    '{{"relevance": "related" or "unrelated", "reason": "one sentence explanation"}}'
)


class ImageChunkGenerator:
    """
    Generates dedicated chunks for images to improve visual content retrieval.

    Each image becomes a separate chunk with:
    - Text content = AI-generated description
    - Metadata includes image URL, dimensions, page info, etc.
    - chunk_type = "image" to distinguish from text chunks

    The fields stored with each image chunk can be configured in the
    application configuration under chunk_fields.image_chunk_fields.
    """

    # Default fields to include in image chunk metadata
    DEFAULT_IMAGE_FIELDS = [
        "chunk_type",
        "image_index",
        "image_url",
        "image_description",
        "image_alt_text",
        "image_type",
        "slide_type",
        "image_page",
        "page_number",
        "image_width",
        "image_height",
        "image_format",
        "image_bbox",
        "document_summary",
    ]

    @classmethod
    def create_image_chunks(
        cls,
        file_id: str,
        image_manifest: Dict[str, Any],
        base_metadata: Dict[str, str],
        page_summary: Optional[str] = None,
        configured_fields: Optional[List[str]] = None,
    ) -> List:
        """
        Create dedicated chunks for each image in the manifest.

        Args:
            file_id: Document file ID
            image_manifest: Manifest dict containing images with descriptions
            base_metadata: Base metadata to include in each chunk
            page_summary: Optional document summary for context
            configured_fields: Optional list of fields to include in metadata.

        Returns:
            List of Chunk objects for images
        """
        from rag_lib.interfaces.chunking_manager import Chunk

        fields_to_include = configured_fields or cls.DEFAULT_IMAGE_FIELDS

        images = image_manifest.get("images", [])
        if not images:
            return []

        image_chunks: List = []

        for idx, img in enumerate(images):
            description = cls._get_description(img)
            if not description or len(description.strip()) < 10:
                continue

            content = cls._build_chunk_content(img, description, base_metadata)
            chunk_metadata = cls._build_chunk_metadata(
                img, base_metadata, page_summary, idx, fields_to_include
            )

            chunk_id = f"{file_id}-img-{idx:03d}"
            chunk = Chunk(
                chunk_id=chunk_id,
                file_id=file_id,
                content=content,
                order=1000 + idx,
                metadata=chunk_metadata,
                parent_id=None,
                level="image",
            )

            image_chunks.append(chunk)

            LOGGER.log_stage(
                ProcessLogger.STAGE_STORAGE,
                f"Created image chunk {chunk_id}: {description[:50]}...",
            )

        return image_chunks

    # ------------------------------------------------------------------
    # Accessors (work with both dataclass objects and plain dicts)
    # ------------------------------------------------------------------

    @classmethod
    def _get_description(cls, img) -> Optional[str]:
        if hasattr(img, "description"):
            return img.description
        elif isinstance(img, dict):
            return img.get("description")
        return None

    @classmethod
    def _get_alt_text(cls, img) -> Optional[str]:
        if hasattr(img, "alt_text"):
            return img.alt_text
        elif isinstance(img, dict):
            return img.get("alt_text")
        return None

    @classmethod
    def _get_page_number(cls, img) -> Optional[int]:
        """Extract page number from image dict/object.

        Tries multiple field names to handle different source formats:
        - page_number: Standard name (used by embeddings adapter)
        - page: PPTX format (from manifest)
        - slide_number: Alternative name
        """
        if hasattr(img, "page_number"):
            return img.page_number
        elif isinstance(img, dict):
            # Try common field name variants
            for field_name in ["page_number", "page", "slide_number"]:
                value = img.get(field_name)
                if value is not None:
                    return value
        return None

    @classmethod
    def _get_storage_path(cls, img) -> Optional[str]:
        if hasattr(img, "storage_path"):
            return img.storage_path
        elif isinstance(img, dict):
            return img.get("storage_path") or img.get("storage_location")
        return None

    @classmethod
    def _get_dimensions(cls, img) -> tuple:
        if hasattr(img, "width") and hasattr(img, "height"):
            return img.width, img.height
        elif isinstance(img, dict):
            return img.get("width"), img.get("height")
        return None, None

    @classmethod
    def _get_format(cls, img) -> Optional[str]:
        if hasattr(img, "format"):
            return img.format
        elif isinstance(img, dict):
            return img.get("format")
        return None

    @classmethod
    def _get_bounding_box(cls, img) -> Optional[Dict]:
        if hasattr(img, "bounding_box"):
            return img.bounding_box
        elif isinstance(img, dict):
            return img.get("bounding_box")
        return None

    @classmethod
    def _get_is_infographic(cls, img) -> bool:
        """Return True if image was classified as an infographic."""
        if hasattr(img, "is_infographic"):
            return bool(img.is_infographic)
        elif isinstance(img, dict):
            return bool(img.get("is_infographic", False))
        return False

    @classmethod
    def _get_context_text(cls, img) -> Optional[str]:
        """Return surrounding document text captured during manifest enrichment."""
        if hasattr(img, "context_text"):
            return img.context_text
        elif isinstance(img, dict):
            return img.get("context_text")
        return None

    # ------------------------------------------------------------------
    # Content / metadata builders
    # ------------------------------------------------------------------

    @classmethod
    def _build_chunk_content(
        cls,
        img,
        description: str,
        base_metadata: Dict[str, str],
    ) -> str:
        parts = ["[Image Content]", f"Description: {description}"]

        # Include surrounding document text for search relevance
        context_text = cls._get_context_text(img)
        if context_text:
            parts.append(f"Context: {context_text}")

        alt_text = cls._get_alt_text(img)
        if alt_text and alt_text.lower() != description.lower()[: len(alt_text)]:
            parts.append(f"Alt text: {alt_text}")

        page_num = cls._get_page_number(img)
        if page_num:
            parts.append(f"Location: Page {page_num}")

        file_name = base_metadata.get("file_name", "")
        if file_name:
            parts.append(f"From document: {file_name}")

        return "\n".join(parts)

    @classmethod
    def _build_chunk_metadata(
        cls,
        img,
        base_metadata: Dict[str, str],
        page_summary: Optional[str],
        image_index: int,
        configured_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metadata = {**base_metadata}
        fields = configured_fields or cls.DEFAULT_IMAGE_FIELDS

        if "chunk_type" in fields:
            metadata["chunk_type"] = "image"
        if "image_index" in fields:
            metadata["image_index"] = str(image_index)

        if "image_url" in fields:
            storage_path = cls._get_storage_path(img)
            if storage_path:
                metadata["image_url"] = storage_path

        if "image_description" in fields:
            description = cls._get_description(img)
            if description:
                metadata["image_description"] = description[:500]

        if "image_type" in fields:
            img_type = img.get("image_type") if isinstance(img, dict) else getattr(img, "image_type", None)
            if img_type:
                metadata["image_type"] = str(img_type)

        if "slide_type" in fields:
            slide_type = img.get("slide_type") if isinstance(img, dict) else getattr(img, "slide_type", None)
            if slide_type:
                metadata["slide_type"] = str(slide_type)

        if "image_alt_text" in fields:
            alt_text = cls._get_alt_text(img)
            if alt_text:
                metadata["image_alt_text"] = alt_text

        if "page_number" in fields or "image_page" in fields:
            page_num = cls._get_page_number(img)

            # If not found in image, try parent chunk metadata (base_metadata)
            if page_num is None and base_metadata:
                page_num = base_metadata.get("page_number")

            if page_num is not None:
                page_num_str = str(page_num)
                if "page_number" in fields:
                    metadata["page_number"] = page_num_str
                if "image_page" in fields:
                    metadata["image_page"] = page_num_str

        if "image_width" in fields or "image_height" in fields:
            width, height = cls._get_dimensions(img)
            if width and height:
                if "image_width" in fields:
                    metadata["image_width"] = str(width)
                if "image_height" in fields:
                    metadata["image_height"] = str(height)

        if "image_format" in fields:
            img_format = cls._get_format(img)
            if img_format:
                metadata["image_format"] = img_format

        if "image_bbox" in fields:
            bbox = cls._get_bounding_box(img)
            if bbox:
                metadata["image_bbox"] = str(bbox)

        if "document_summary" in fields and page_summary:
            metadata["document_summary"] = page_summary

        # Infographic classification flag (from Vision LLM description step)
        is_infographic = cls._get_is_infographic(img)
        metadata["infographic"] = "true" if is_infographic else "false"

        # Unified media envelope for this image chunk
        media_entry = {}
        storage_path = cls._get_storage_path(img)
        if storage_path:
            media_entry["url"] = storage_path
        img_type = img.get("image_type") if isinstance(img, dict) else getattr(img, "image_type", None)
        if img_type:
            media_entry["image_type"] = str(img_type)
        desc = cls._get_description(img)
        if desc:
            media_entry["description"] = desc[:500]
        page_num = cls._get_page_number(img)
        if page_num:
            media_entry["page"] = str(page_num)
        w, h = cls._get_dimensions(img)
        if w:
            media_entry["width"] = str(w)
        if h:
            media_entry["height"] = str(h)

        if is_infographic:
            media = {"infographics": [media_entry], "images": [], "videos": []}
        else:
            media = {"infographics": [], "images": [media_entry], "videos": []}
        metadata["media"] = json.dumps(media)
        metadata["has_media"] = "true"

        return metadata

    # ------------------------------------------------------------------
    # Infographic page chunk creation
    # ------------------------------------------------------------------

    @classmethod
    def create_infographic_page_chunks(
        cls,
        file_id: str,
        file_path: str,
        page_numbers: List[int],
        image_repository,
        base_metadata: Dict[str, str],
        *,
        region: Optional[str] = None,
        vision_model: Optional[str] = None,
        dpi: int = 200,
    ) -> List:
        """Create chunks for infographic / visually-rich PDF pages.

        For each specified page this method:
        1. Renders the page to a PNG (via pypdfium2).
        2. Uploads the PNG to object storage.
        3. Calls a Vision LLM for a detailed description.
        4. Creates **two** :class:`Chunk` objects per page:
           - A *text-embedded* chunk (``image_base64=None``): the Vision LLM
             description is used as ``embedding_content`` so that text queries
             like "revenue of Wipro" match this page.
           - An *image-embedded* chunk (``image_base64`` set): the raw PNG
             base64 is attached so the Cohere embed-v4 adapter generates a
             multimodal embedding.
           Both chunks point to the same object storage image via ``image_url`` metadata.

        Args:
            file_id: Document identifier.
            file_path: Local path to the PDF file.
            page_numbers: 1-based page numbers to process.
            image_repository: :class:`ImageRepository` for uploading to object storage.
            base_metadata: Base metadata dict to copy into each chunk.
            region: Deprecated compatibility argument; cloud-lib resolves location.
            vision_model: Optional concrete LLM model override.
            dpi: Resolution for page rendering (default 200).

        Returns:
            List of Chunk objects (2 per infographic page).
        """
        import pypdfium2 as pdfium

        from rag_lib.interfaces.chunking_manager import Chunk
        from services.llm import LLMConverseClient

        # Import the shared pypdfium2 lock (PDFium C library is not thread-safe)
        try:
            from docling.utils.locks import pypdfium2_lock as _pdfium_lock
        except ImportError:
            import threading
            _pdfium_lock = threading.Lock()

        converse = LLMConverseClient(model_key="vision")

        # Render all pages under the lock, then release it for VLM calls
        rendered_pages: Dict[int, tuple] = {}  # page_num → (png_bytes, b64_png)
        scale = dpi / 72  # pypdfium2 scale factor: 1.0 = 72 DPI
        with _pdfium_lock:
            pdf = pdfium.PdfDocument(file_path)
            try:
                for page_num in page_numbers:
                    page_idx = page_num - 1
                    if page_idx < 0 or page_idx >= len(pdf):
                        LOGGER.log_warning(
                            ProcessLogger.STAGE_CHUNK,
                            f"Page {page_num} out of range for {file_path} (total pages: {len(pdf)}), skipping",
                        )
                        continue
                    page = pdf[page_idx]
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG", optimize=True)
                    png_bytes = buf.getvalue()
                    b64_png = base64.b64encode(png_bytes).decode("utf-8")
                    rendered_pages[page_num] = (png_bytes, b64_png)
            finally:
                pdf.close()

        chunks: List = []

        for page_num, (png_bytes, b64_png) in rendered_pages.items():
            try:
                # 2) Upload to object storage
                image_id = f"infographic-page-{page_num:03d}"
                storage_uri = image_repository.put_image(
                    document_id=file_id,
                    image_id=image_id,
                    image_bytes=png_bytes,
                    image_format="png",
                )

                # 3) Vision LLM description
                vlm_bytes, vlm_fmt = _compress_for_vlm(png_bytes, page_num)
                try:
                    description = converse.converse(
                        model_id=vision_model,
                        prompt=_INFOGRAPHIC_DESCRIPTION_PROMPT,
                        image_bytes=vlm_bytes,
                        image_format=vlm_fmt,
                        max_tokens=600,
                        temperature=0.1,
                    )
                    description = (description or "").strip()
                except Exception as e:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_CHUNK,
                        f"Vision LLM failed for page {page_num}: {e}",
                    )
                    description = f"Infographic page {page_num} from {base_metadata.get('file_name', 'document')}"

                # Build media envelope for this infographic page
                _info_entry = {
                    "url": storage_uri,
                    "description": description[:500],
                    "page": str(page_num),
                }
                _media_json = json.dumps(
                    {"infographics": [_info_entry], "images": [], "videos": []}
                )

                # 4a) Text-embedded chunk (for text-query retrieval)
                text_meta = {
                    **base_metadata,
                    "chunk_type": "infographic",
                    "infographic_page": str(page_num),
                    "image_url": storage_uri,
                    "image_description": description[:500],
                    "page_number": str(page_num),
                    "infographic": "true",
                    "has_media": "true",
                    "media": _media_json,
                }
                text_chunk = Chunk(
                    chunk_id=f"{file_id}-infographic-text-p{page_num:03d}",
                    file_id=file_id,
                    content=f"[Infographic Page {page_num}]\n{description}",
                    order=2000 + page_num,
                    metadata=text_meta,
                    parent_id=None,
                    level="infographic",
                    embedding_content=description,
                )
                chunks.append(text_chunk)

                # 4b) Image-embedded chunk (for image-query / cross-modal retrieval)
                img_meta = {
                    **base_metadata,
                    "chunk_type": "infographic_image",
                    "infographic_page": str(page_num),
                    "image_url": storage_uri,
                    "image_description": description[:500],
                    "page_number": str(page_num),
                    "infographic": "true",
                    "has_media": "true",
                    "media": _media_json,
                }
                img_chunk = Chunk(
                    chunk_id=f"{file_id}-infographic-img-p{page_num:03d}",
                    file_id=file_id,
                    content=f"[Infographic Page {page_num}]\n{description}",
                    order=3000 + page_num,
                    metadata=img_meta,
                    parent_id=None,
                    level="infographic",
                    embedding_content=None,
                    image_base64=b64_png,
                )
                chunks.append(img_chunk)

                LOGGER.log_stage(
                    ProcessLogger.STAGE_CHUNK,
                    f"Created infographic chunks for page {page_num}: {description[:60]}...",
                )

            except Exception as e:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_CHUNK,
                    f"Failed to create infographic chunk for page {page_num}: {e}",
                )

        return chunks

    # ------------------------------------------------------------------
    # Vision LLM relevance tagging
    # ------------------------------------------------------------------

    @classmethod
    def tag_image_relevance(
        cls,
        images: List[Dict[str, Any]],
        chunks: List,
        *,
        region: Optional[str] = None,
        vision_model: Optional[str] = None,
        max_workers: int = 5,
    ) -> List[Dict[str, Any]]:
        """Tag each image as ``related`` or ``unrelated`` using a Vision LLM.

        For each image the method finds the nearest text chunk (by page number)
        and asks the Vision LLM whether the image is content-relevant given
        that surrounding text context.

        The ``relevance`` and ``relevance_reason`` keys are added **in-place**
        to each image dict and also returned.

        .. note::
           All images are kept regardless of relevance.  Down-stream consumers
           (e.g. the chat display layer) can filter by ``relevance == "related"``
           to show only meaningful images by default.

        Args:
            images: List of image dicts (from ``image_manifest["images"]``).
                Each must have ``storage_path`` and optionally ``image_bytes``
                or ``page_number``.
            chunks: List of text Chunk objects from the same document (used to
                extract surrounding text context).
            region: Deprecated compatibility argument; cloud-lib resolves location.
            vision_model: Optional concrete LLM model override.
            max_workers: Parallel workers for LLM calls.

        Returns:
            The same ``images`` list with ``relevance`` and ``relevance_reason``
            added to each entry.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from services.llm import LLMConverseClient

        if not images:
            return images

        converse = LLMConverseClient(model_key="vision")

        # Build a page → text lookup from chunks
        page_text: Dict[int, str] = {}
        for c in chunks:
            pn = c.metadata.get("page_number") or c.metadata.get("page")
            if pn:
                try:
                    pn_int = int(pn)
                    existing = page_text.get(pn_int, "")
                    page_text[pn_int] = (existing + "\n" + c.content).strip()
                except (ValueError, TypeError):
                    pass

        def _tag_single(img: Dict[str, Any]) -> None:
            """Tag a single image dict in-place."""
            try:
                img_bytes = img.get("image_bytes")
                if not img_bytes and img.get("_image_bytes_ref"):
                    img_bytes = img["_image_bytes_ref"]

                if not img_bytes:
                    # Cannot call vision without image bytes — default to related
                    img["relevance"] = "related"
                    img["relevance_reason"] = "No image bytes available for analysis"
                    return

                # Find surrounding text context
                page_num = img.get("page_number")
                context = ""
                if page_num:
                    context = page_text.get(int(page_num), "")[:1500]
                if not context:
                    context = "(No surrounding text available)"

                prompt = _RELEVANCE_TAGGING_PROMPT.format(context=context)

                img_format = (img.get("format") or "png").lower()
                if img_format not in ("png", "jpeg", "gif", "webp"):
                    img_format = "png"

                raw = converse.converse(
                    model_id=vision_model,
                    prompt=prompt,
                    image_bytes=img_bytes if isinstance(img_bytes, bytes) else base64.b64decode(img_bytes),
                    image_format=img_format,
                    max_tokens=150,
                    temperature=0.0,
                )

                # Parse JSON response
                from rag_lib.utils.json_utils import parse_llm_json

                result = parse_llm_json((raw or "").strip())
                if result is None:
                    raise ValueError(f"Failed to parse JSON from vision response: {(raw or '')[:80]}")
                img["relevance"] = result.get("relevance", "related")
                img["relevance_reason"] = result.get("reason", "")
            except Exception as e:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_CHUNK,
                    f"Relevance tagging failed for image {img.get('image_id', '?')}: {e}",
                )
                img["relevance"] = "related"  # default to related on failure
                img["relevance_reason"] = f"Tagging failed: {e}"

        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Tagging relevance for {len(images)} images with Vision LLM",
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_tag_single, img) for img in images]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        related = sum(1 for img in images if img.get("relevance") == "related")
        LOGGER.log_stage(
            ProcessLogger.STAGE_CHUNK,
            f"Relevance tagging complete: {related}/{len(images)} images tagged as related",
        )

        return images

    # ------------------------------------------------------------------
    # Web / HTML infographic chunk creation
    # ------------------------------------------------------------------

    @classmethod
    def create_web_infographic_chunks(
        cls,
        file_id: str,
        image_manifest: Dict[str, Any],
        base_metadata: Dict[str, str],
    ) -> List:
        """Create infographic chunks for web/HTML images classified as infographics.

        Similar to :meth:`create_infographic_page_chunks` but works from
        **already-extracted and stored** images (no PDF rendering needed).

        For each infographic image two :class:`Chunk` objects are created:
        - A *text-embedded* chunk (``image_base64=None``): the description is
          used as ``embedding_content`` for text-query retrieval.
        - An *image-embedded* chunk (``image_base64`` set): raw image bytes
          are base64-encoded for multimodal embedding (Cohere embed-v4).

        Args:
            file_id: Document identifier.
            image_manifest: Image manifest dict with ``"images"`` list.
                Each image must have ``storage_path``, ``description``,
                and ``is_infographic`` set to ``True``.
            base_metadata: Base metadata dict to copy into each chunk.

        Returns:
            List of Chunk objects (2 per infographic image).
        """
        from rag_lib.interfaces.chunking_manager import Chunk

        images = image_manifest.get("images", [])
        if not images:
            return []

        chunks: List = []

        for img in images:
            # Only process images flagged as infographics
            is_info = cls._get_is_infographic(img)
            if not is_info:
                continue

            description = cls._get_description(img)
            if not description or len(description.strip()) < 10:
                continue

            storage_uri = cls._get_storage_path(img)
            image_id = img.get("image_id") if isinstance(img, dict) else getattr(img, "image_id", None)

            # Get image bytes for multimodal embedding
            img_bytes = img.get("image_bytes") if isinstance(img, dict) else getattr(img, "image_bytes", None)
            b64_img = base64.b64encode(img_bytes).decode("utf-8") if img_bytes else None

            img_format = cls._get_format(img) or "png"
            page_num = cls._get_page_number(img)

            # Build media envelope for this web infographic
            _info_entry: Dict[str, str] = {}
            if storage_uri:
                _info_entry["url"] = storage_uri
            if description:
                _info_entry["description"] = description[:500]
            if page_num:
                _info_entry["page"] = str(page_num)
            _media_json = json.dumps(
                {"infographics": [_info_entry], "images": [], "videos": []}
            )

            # 1) Text-embedded chunk (for text-query retrieval)
            text_meta = {
                **base_metadata,
                "chunk_type": "infographic",
                "image_url": storage_uri or "",
                "image_description": description[:500],
                "infographic": "true",
                "has_media": "true",
                "media": _media_json,
            }
            if page_num:
                text_meta["page_number"] = str(page_num)
                text_meta["infographic_page"] = str(page_num)

            text_chunk = Chunk(
                chunk_id=f"{file_id}-infographic-text-{image_id or 'img'}",
                file_id=file_id,
                content=f"[Infographic Image]\n{description}",
                order=2000 + (page_num or 0),
                metadata=text_meta,
                parent_id=None,
                level="infographic",
                embedding_content=description,
            )
            chunks.append(text_chunk)

            # 2) Image-embedded chunk (for cross-modal retrieval)
            if b64_img:
                img_meta = {
                    **base_metadata,
                    "chunk_type": "infographic_image",
                    "image_url": storage_uri or "",
                    "image_description": description[:500],
                    "infographic": "true",
                    "has_media": "true",
                    "media": _media_json,
                }
                if page_num:
                    img_meta["page_number"] = str(page_num)
                    img_meta["infographic_page"] = str(page_num)

                img_chunk = Chunk(
                    chunk_id=f"{file_id}-infographic-img-{image_id or 'img'}",
                    file_id=file_id,
                    content=f"[Infographic Image]\n{description}",
                    order=3000 + (page_num or 0),
                    metadata=img_meta,
                    parent_id=None,
                    level="infographic",
                    embedding_content=None,
                    image_base64=b64_img,
                )
                chunks.append(img_chunk)

            LOGGER.log_stage(
                ProcessLogger.STAGE_CHUNK,
                f"Created web infographic chunks for image {image_id}: {description[:60]}...",
            )

        return chunks
