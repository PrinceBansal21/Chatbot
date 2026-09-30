"""Vision-to-text service for describing document images."""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from utils.logger import get_logger, ProcessLogger
from .llm_converse_client import LLMConverseClient

LOGGER = get_logger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMAGE_COMMENT_PATTERN = re.compile(r"<!--\s*image(?P<meta>.*?)-->", re.IGNORECASE | re.DOTALL)
_DATA_URI_PATTERN = re.compile(r"data:image/(?P<fmt>[^;]+);base64,(?P<data>.+)")
_ALLOWED_FORMATS = {"jpeg", "jpg", "png", "gif", "webp"}


@dataclass
class _ImageMatch:
    start: int
    end: int
    alt_text: str
    source: Optional[str]
    image_bytes: Optional[bytes]
    image_format: Optional[str]


class _HtmlImageExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: List[Dict[str, Optional[str]]] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        src = attr_map.get("src")
        alt = attr_map.get("alt")
        if src:
            self.images.append({"src": src, "alt": alt})


class ImageDescriptionAugmenter:
    """Generates textual descriptions for images referenced inside Markdown."""

    DEFAULT_PROMPT = (
        "You generate helpful descriptions for document images to improve search. "
        "Describe the key objects, text, and context in two concise sentences. "
        "If provided, consider the existing alt text: {alt_text}."
    )

    def __init__(self, config: Dict):
        self.config = config.copy()
        self.default_provider = self.config.get("default_provider", "cloud")
        self._managers = {}
        self._configs = {}
        self._clients = {}

    def _get_config(self, provider: str):
        if provider not in self._configs:
            if provider != "cloud":
                raise ValueError(
                    f"Unsupported vision provider {provider!r}; expected 'cloud'."
                )
            provider_config = self.config.get("cloud", {}) or self.config
            self._configs[provider] = {
                "prompt_template": provider_config.get("prompt") or self.DEFAULT_PROMPT,
                "max_images": int(provider_config.get("max_images", 50)),
                "max_tokens": int(provider_config.get("max_tokens", 256)),
                "temperature": float(provider_config.get("temperature", 0.2)),
                "http_timeout": float(provider_config.get("http_timeout", 5.0)),
                "max_download_bytes": int(provider_config.get("max_download_bytes", 5 * 1024 * 1024)),
                "user_agent": provider_config.get("user_agent", "data-ingestion-pipeline/vision/1.0")
            }
        return self._configs[provider]

    def _get_client(self, provider: str):
        if provider not in self._clients:
            self._clients[provider] = LLMConverseClient(model_key="vision")
        return self._clients[provider]

    def annotate_markdown(
        self,
        markdown: str,
        *,
        asset_root: Optional[str],
        file_id: Optional[str],
        source_path: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """Find images inside Markdown or Docling placeholders, then append descriptions."""
        
        provider = provider or self.default_provider
        config = self._get_config(provider)
        
        # Temporarily attach config to self for helper methods (refactor later to pass config)
        self.max_images = config["max_images"]
        self.max_download_bytes = config["max_download_bytes"]
        self.http_timeout = config["http_timeout"]
        self.user_agent = config["user_agent"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.prompt_template = config["prompt_template"]
        self._client = self._get_client(provider)

        matches = self._extract_markdown_images(markdown, asset_root)
        remaining = self.max_images - len(matches)
        if remaining > 0:
            comments = self._extract_comment_placeholders(markdown, remaining)
            if comments:
                self._hydrate_comment_sources(comments, asset_root, source_path, markdown)
                matches.extend(comments)

        matches.sort(key=lambda item: item.start)
        describe_targets = [match for match in matches if match.image_bytes]
        if not describe_targets:
            return markdown

        descriptions = self._describe_images(describe_targets, file_id)
        description_iter = iter(descriptions)
        cursor = 0
        parts: List[str] = []
        for match in matches:
            parts.append(markdown[cursor : match.end])
            if match.image_bytes:
                description = next(description_iter)
                if description:
                    parts.append(f"\n\n> Image description: {description.strip()}\n")
            cursor = match.end
        parts.append(markdown[cursor:])
        return "".join(parts)

    def _extract_markdown_images(self, markdown: str, asset_root: Optional[str]) -> List[_ImageMatch]:
        results: List[_ImageMatch] = []
        for match in _IMAGE_PATTERN.finditer(markdown):
            if len(results) >= self.max_images:
                break
            src = match.group("src")
            resolved = self._resolve_image_source(src, asset_root, base_path=None)
            if not resolved:
                continue
            results.append(
                _ImageMatch(
                    start=match.start(),
                    end=match.end(),
                    alt_text=match.group("alt"),
                    source=src,
                    image_bytes=resolved["bytes"],
                    image_format=resolved["format"],
                )
            )
        return results

    def _extract_comment_placeholders(self, markdown: str, limit: int) -> List[_ImageMatch]:
        matches: List[_ImageMatch] = []
        for comment in _IMAGE_COMMENT_PATTERN.finditer(markdown):
            if len(matches) >= limit:
                break
            meta = (comment.group("meta") or "").strip()
            source_hint: Optional[str] = None
            alt_text = ""
            if meta:
                cleaned = meta.strip("- ")
                if cleaned:
                    try:
                        payload = json.loads(cleaned)
                        source_hint = payload.get("src") or payload.get("path")
                        alt_text = payload.get("alt") or payload.get("description") or ""
                    except json.JSONDecodeError:
                        if ":" in cleaned:
                            source_hint = cleaned.split(":", 1)[1].strip()
                        else:
                            source_hint = cleaned
            matches.append(
                _ImageMatch(
                    start=comment.start(),
                    end=comment.end(),
                    alt_text=alt_text,
                    source=source_hint,
                    image_bytes=None,
                    image_format=None,
                )
            )
        return matches

    def _hydrate_comment_sources(
        self,
        placeholders: List[_ImageMatch],
        asset_root: Optional[str],
        source_path: Optional[str],
        markdown: str = "",
    ) -> None:
        if not placeholders:
            return
        
        # Check if source is a PDF - extract images from PDF pages
        if source_path and source_path.lower().endswith(".pdf"):
            self._hydrate_pdf_images(placeholders, source_path, markdown)
            return
        
        # For HTML sources, use existing logic
        html_sources = self._extract_html_images(source_path)
        html_idx = 0
        for match in placeholders:
            resolved = None
            if match.source:
                resolved = self._resolve_image_source(match.source, asset_root, base_path=source_path)
            while not resolved and html_idx < len(html_sources):
                candidate = html_sources[html_idx]
                html_idx += 1
                resolved = self._resolve_image_source(candidate.get("src"), asset_root, base_path=source_path)
                if resolved:
                    match.source = candidate.get("src")
                    if not match.alt_text:
                        match.alt_text = candidate.get("alt") or ""
            if resolved:
                match.image_bytes = resolved["bytes"]
                match.image_format = resolved["format"]

    def _hydrate_pdf_images(
        self,
        placeholders: List[_ImageMatch],
        pdf_path: str,
        markdown: str = "",
    ) -> None:
        """
        Extract images from PDF to match with <!-- image --> placeholders.
        
        Strategy:
        1. Parse markdown to find page numbers for each placeholder
        2. For each placeholder, use its page context to extract/render the right image
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            LOGGER.log_warning(
                ProcessLogger.STAGE_VISION,
                "pypdfium2 not available for PDF image extraction"
            )
            return
        
        import re
        page_pattern = re.compile(r'<!-- page_number: (\d+) -->')
        page_markers = [(m.start(), int(m.group(1))) for m in page_pattern.finditer(markdown)]
        
        def get_page_for_position(pos: int) -> int:
            current_page = 1
            for marker_pos, page_num in page_markers:
                if marker_pos > pos:
                    break
                current_page = page_num
            return current_page
        
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            
            # Extract embedded images from all pages using pypdfium2 object iteration
            embedded_images_by_page: dict = {}
            for page_num_idx in range(total_pages):
                page = pdf[page_num_idx]
                page_images = []
                img_idx = 0
                for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
                    try:
                        pil_img = obj.get_bitmap().to_pil()
                        width, height = pil_img.size
                        if width >= 50 and height >= 50:
                            buf = BytesIO()
                            pil_img.save(buf, format="PNG")
                            page_images.append({
                                "page": page_num_idx + 1,
                                "bytes": buf.getvalue(),
                                "format": "png",
                                "width": width,
                                "height": height,
                                "index": img_idx,
                            })
                    except Exception:
                        pass
                    img_idx += 1
                if page_images:
                    embedded_images_by_page[page_num_idx + 1] = page_images
            
            total_embedded = sum(len(imgs) for imgs in embedded_images_by_page.values())
            LOGGER.log_debug(
                ProcessLogger.STAGE_VISION,
                f"Found {total_embedded} embedded images across {len(embedded_images_by_page)} pages for {len(placeholders)} placeholders"
            )
            
            dpi = 150
            scale = dpi / 72
            used_images: set = set()
            
            for idx, match in enumerate(placeholders):
                placeholder_page = get_page_for_position(match.start)
                
                page_images = embedded_images_by_page.get(placeholder_page, [])
                large_image = None
                for img in page_images:
                    img_key = (img["page"], img["index"])
                    if img_key in used_images:
                        continue
                    if img["width"] >= 200 and img["height"] >= 200:
                        large_image = img
                        used_images.add(img_key)
                        break
                
                if large_image:
                    match.image_bytes = large_image["bytes"]
                    match.image_format = large_image["format"]
                    match.source = f"pdf_page_{large_image['page']}_embedded"
                    LOGGER.log_debug(
                        ProcessLogger.STAGE_VISION,
                        f"Placeholder {idx + 1} (page {placeholder_page}): Using embedded image ({large_image['width']}x{large_image['height']})"
                    )
                else:
                    page_idx = min(placeholder_page - 1, total_pages - 1)
                    page_idx = max(0, page_idx)
                    
                    page = pdf[page_idx]
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    
                    match.image_bytes = buf.getvalue()
                    match.image_format = "png"
                    match.source = f"pdf_page_{page_idx + 1}_rendered"
                    
                    LOGGER.log_debug(
                        ProcessLogger.STAGE_VISION,
                        f"Placeholder {idx + 1} (page {placeholder_page}): Rendered page {page_idx + 1} (vector graphics/chart)"
                    )
            
            pdf.close()
                    
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_VISION,
                f"Failed to extract images from PDF: {e}"
            )

    def _extract_html_images(self, source_path: Optional[str]) -> List[Dict[str, Optional[str]]]:
        if not source_path or not source_path.lower().endswith((".html", ".htm")):
            return []
        try:
            html_text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            LOGGER.log_debug(ProcessLogger.STAGE_VISION, f"Unable to read source file for image mapping: {source_path}")
            return []
        parser = _HtmlImageExtractor()
        parser.feed(html_text)
        return parser.images

    def _resolve_image_source(
        self,
        src: Optional[str],
        asset_root: Optional[str],
        *,
        base_path: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not src:
            return None
        sanitized = src.strip().strip('"').strip("'")
        if not sanitized:
            return None
        data_match = _DATA_URI_PATTERN.match(sanitized)
        if data_match:
            try:
                data = base64.b64decode(data_match.group("data"), validate=True)
            except Exception:
                LOGGER.log_warning(ProcessLogger.STAGE_VISION, f"Failed to decode inline image data; src prefix={sanitized[:32]}")
                return None
            detected = self._detect_image_format(data)
            image_format = self._select_allowed_format(detected, data_match.group("fmt"))
            if not image_format:
                LOGGER.log_debug(ProcessLogger.STAGE_VISION, f"Skipping inline image; unsupported format: {data_match.group('fmt')}")
                return None
            return {"bytes": data, "format": image_format}

        if self._is_http_url(sanitized):
            return self._download_remote_image(sanitized)

        candidates: List[Path] = []
        if os.path.isabs(sanitized):
            candidates.append(Path(sanitized))
        if asset_root:
            candidates.append(Path(asset_root) / sanitized)
        elif not os.path.isabs(sanitized):
            candidates.append(Path(sanitized))

        for candidate in candidates:
            if candidate.is_file():
                fmt = candidate.suffix.replace(".", "").lower() or "png"
                try:
                    data = candidate.read_bytes()
                    detected = self._detect_image_format(data)
                    image_format = self._select_allowed_format(detected, fmt)
                    if not image_format:
                        LOGGER.log_debug(
                            ProcessLogger.STAGE_VISION,
                            f"Skipping image {candidate}; unsupported format: {detected or fmt}"
                        )
                        continue
                    return {"bytes": data, "format": image_format}
                except OSError:
                    LOGGER.log_warning(ProcessLogger.STAGE_VISION, f"Unable to read image asset: {candidate}")
                    return None

        if base_path and self._is_http_url(base_path):
            joined = urljoin(base_path, sanitized)
            return self._download_remote_image(joined)

        return None

    def _download_remote_image(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.http_timeout) as response:  # nosec - controlled URL
                data = response.read(self.max_download_bytes + 1)
                if len(data) > self.max_download_bytes:
                    LOGGER.log_warning(ProcessLogger.STAGE_VISION, f"Remote image exceeds max download size ({self.max_download_bytes} bytes): {url}")
                    return None
                content_type = response.headers.get("Content-Type", "")
        except Exception as e:
            LOGGER.log_warning(ProcessLogger.STAGE_VISION, f"Failed to download remote image: {url}", extra_data={"error": str(e)})
            return None
        fmt = self._infer_format_from_mime(content_type, url)
        detected = self._detect_image_format(data)
        image_format = self._select_allowed_format(detected, fmt)
        if not image_format:
            LOGGER.log_debug(ProcessLogger.STAGE_VISION, f"Skipping remote image; unsupported format {detected or fmt}: {url}")
            return None
        return {"bytes": data, "format": image_format}

    @staticmethod
    def _infer_format_from_mime(content_type: str, fallback_url: Optional[str]) -> str:
        if content_type and "/" in content_type:
            subtype = content_type.split("/", 1)[1]
            return (subtype.split("+")[0]).lower()
        if fallback_url:
            parsed = urlparse(fallback_url)
            ext = Path(parsed.path).suffix.replace(".", "").lower()
            if ext:
                return ext
        return "png"

    @staticmethod
    def _is_http_url(value: str) -> bool:
        return value.lower().startswith(("http://", "https://"))

    @staticmethod
    def _select_allowed_format(*candidates: Optional[str]) -> Optional[str]:
        for candidate in candidates:
            if not candidate:
                continue
            normalized = candidate.lower()
            if normalized in _ALLOWED_FORMATS:
                return "jpeg" if normalized == "jpg" else normalized
        return None

    def _describe_images(self, matches: List[_ImageMatch], file_id: Optional[str]) -> List[Optional[str]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        prompt = self.prompt_template.format(alt_text="{alt_text}", file_id=file_id or "")

        def _describe_one(idx: int, match: _ImageMatch) -> tuple:
            """Return (idx, description_or_None) for a single image."""
            try:
                LOGGER.log_debug(ProcessLogger.STAGE_VISION, f"Describing image {idx + 1}/{len(matches)}: {match.source}")

                image_format = (match.image_format or "png").lower()
                if image_format not in _ALLOWED_FORMATS and image_format != "jpg":
                    if match.alt_text:
                        LOGGER.log_debug(
                            ProcessLogger.STAGE_VISION,
                            f"Image format '{image_format}' not supported by Vision LLM, using alt text as description"
                        )
                        return idx, f"[Alt text] {match.alt_text}"
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_VISION,
                        f"Image format '{image_format}' not supported and no alt text available for: {match.source}"
                    )
                    return idx, None

                if not match.image_bytes:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_VISION,
                        f"Image {idx + 1}: No image bytes available for {match.source}, using alt text fallback"
                    )
                    return idx, (f"[Alt text] {match.alt_text}" if match.alt_text else None)

                img_prompt = prompt.format(alt_text=match.alt_text or "")
                description = self._invoke_vlm(img_prompt, match)
                return idx, (description.strip() if description else None)
            except Exception as e:
                LOGGER.log_error(
                    f"Failed to describe image: {match.source} - {type(e).__name__}: {str(e)}",
                    exception=e,
                    extra_data={
                        "image_source": match.source,
                        "image_format": match.image_format,
                        "has_bytes": match.image_bytes is not None,
                        "bytes_length": len(match.image_bytes) if match.image_bytes else 0,
                        "model_key": "vision",
                    },
                )
                if match.alt_text:
                    LOGGER.log_debug(ProcessLogger.STAGE_VISION, "Using alt text as fallback due to error")
                    return idx, f"[Alt text] {match.alt_text}"
                return idx, None

        # Parallel VLM calls — IO-bound (LLM API), safe to fan out
        max_workers = min(len(matches), 5)
        descriptions: List[Optional[str]] = [None] * len(matches)

        if len(matches) == 1:
            _, desc = _describe_one(0, matches[0])
            descriptions[0] = desc
        else:
            LOGGER.log_debug(
                ProcessLogger.STAGE_VISION,
                f"Describing {len(matches)} images in parallel (max_workers={max_workers})",
            )
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="img_desc") as pool:
                futures = {pool.submit(_describe_one, i, m): i for i, m in enumerate(matches)}
                for future in as_completed(futures):
                    i, desc = future.result()
                    descriptions[i] = desc

        return descriptions

    def _invoke_vlm(self, prompt: str, match: _ImageMatch) -> Optional[str]:
        return self._client.converse(
            prompt=prompt,
            image_bytes=match.image_bytes,
            image_format=match.image_format,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _detect_image_format(self, image_bytes: bytes) -> Optional[str]:
        try:
            from PIL import Image  # type: ignore
        except ImportError:  # pragma: no cover
            return None

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                fmt = image.format
                return fmt.lower() if fmt else None
        except Exception:
            return None
