"""Document fetcher service for downloading files from cloud storage and HTTP URLs."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse, unquote

import requests

from cloud_lib.factory import CloudFactory
from cloud_lib.exceptions import CloudlibException

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)

# Content-Type to file extension mapping for HTTP downloads
CONTENT_TYPE_TO_EXT = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "application/json": ".json",
}


@dataclass
class FetchedDocumentBundle:
    """Represents a downloaded document and its supporting assets."""

    document_path: str
    asset_root: str
    cleanup_root: str


class DocumentFetcher:
    """Downloads documents from cloud object storage or HTTP/HTTPS URLs."""

    # HTTP request timeout (connect, read) in seconds
    HTTP_TIMEOUT = (10, 120)
    HTTP_MAX_SIZE = 500 * 1024 * 1024  # 500 MB

    def __init__(self, region: str):
        # Auto-detect cloud provider from CLOUD_PROVIDER env var
        self.file_manager = CloudFactory.get_file_manager(bucket_key="markdown_output")

    def download(self, source_uri: str, file_type: Optional[str] = None) -> FetchedDocumentBundle:
        """Download a document from cloud storage or HTTP/HTTPS URL.
        
        Args:
            source_uri: Cloud object storage URI or HTTP/HTTPS URL
            file_type: Optional explicit file type hint
            
        Returns:
            FetchedDocumentBundle with local path to the downloaded document
        """
        # Route HTTP/HTTPS URLs to the HTTP downloader
        if source_uri.lower().startswith(("http://", "https://")):
            return self._download_http(source_uri, file_type)
        
        # Cloud object storage path
        return self._download_cloud_storage(source_uri, file_type)

    def _download_http(self, url: str, file_type: Optional[str] = None) -> FetchedDocumentBundle:
        """Download a document from an HTTP/HTTPS URL.
        
        Supports both direct document links (PDF, DOCX, etc.) and web pages.
        Determines filename and extension from URL path, Content-Disposition
        header, or Content-Type header.
        
        Args:
            url: HTTP/HTTPS URL to download
            file_type: Optional explicit file type hint
            
        Returns:
            FetchedDocumentBundle with local path to the downloaded file
        """
        tmp_dir = tempfile.mkdtemp(prefix="document_fetcher_http_")
        LOGGER.log_stage(ProcessLogger.STAGE_FETCH, f"Downloading from HTTP URL: {url}")
        
        try:
            response = requests.get(
                url,
                timeout=self.HTTP_TIMEOUT,
                stream=True,
                headers={"User-Agent": "WIN-DataIngestion/1.0"},
                allow_redirects=True,
            )
            response.raise_for_status()
            
            # Determine filename
            filename = self._resolve_http_filename(url, response, file_type)
            local_path = os.path.join(tmp_dir, filename)
            
            # Stream to disk
            content_length = 0
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    content_length += len(chunk)
                    if content_length > self.HTTP_MAX_SIZE:
                        raise ValueError(f"Download exceeds max size ({self.HTTP_MAX_SIZE} bytes): {url}")
                    f.write(chunk)
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"HTTP download complete: {filename} ({content_length:,} bytes)"
            )
            
            return FetchedDocumentBundle(
                document_path=local_path,
                asset_root=tmp_dir,
                cleanup_root=tmp_dir,
            )
            
        except requests.RequestException as e:
            # Clean up temp dir on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise IOError(f"Failed to download from {url}: {e}") from e

    def _resolve_http_filename(
        self, url: str, response: requests.Response, file_type: Optional[str] = None
    ) -> str:
        """Determine the local filename for an HTTP download.
        
        Priority:
        1. Content-Disposition header (server-suggested filename)
        2. URL path basename (if it has a recognizable extension)
        3. Explicit file_type hint + fallback name
        4. Content-Type header mapping
        5. Default: page.html
        """
        # 1. Content-Disposition
        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            # Extract filename from header (handles both quoted and unquoted)
            parts = cd.split("filename=")
            if len(parts) > 1:
                name = parts[-1].strip().strip('"').strip("'")
                if name:
                    return self._sanitize_filename(name)
        
        # 2. URL path
        parsed = urlparse(url)
        url_path = unquote(parsed.path).rstrip("/")
        if url_path:
            basename = os.path.basename(url_path)
            _, ext = os.path.splitext(basename)
            if ext and ext.lower() in (".pdf", ".doc", ".docx", ".xls", ".xlsx",
                                        ".ppt", ".pptx", ".html", ".htm", ".txt",
                                        ".csv", ".md", ".json"):
                return self._sanitize_filename(basename)
        
        # 3. Explicit file_type hint
        if file_type:
            ext = f".{file_type.lower().lstrip('.')}"
            return f"document{ext}"
        
        # 4. Content-Type header
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        ext = CONTENT_TYPE_TO_EXT.get(content_type)
        if ext:
            return f"document{ext}"
        
        # 5. Fallback
        return "page.html"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove unsafe characters from a filename."""
        # Keep alphanumeric, dots, dashes, underscores
        safe = "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
        return safe or "document"

    @staticmethod
    def _safe_join_tmp_dir(tmp_dir: str, relative_path: str) -> str:
        """Join a relative object path under tmp_dir without allowing traversal."""
        normalized = os.path.normpath(relative_path).lstrip(os.sep)
        if normalized in ("", "."):
            raise ValueError("Object key resolved to an empty local path")
        if normalized.startswith("..") or os.path.isabs(relative_path):
            raise ValueError(f"Unsafe object key path: {relative_path}")

        local_path = os.path.abspath(os.path.join(tmp_dir, normalized))
        tmp_root = os.path.abspath(tmp_dir)
        if os.path.commonpath([tmp_root, local_path]) != tmp_root:
            raise ValueError(f"Object key escapes temporary directory: {relative_path}")
        return local_path

    def _download_cloud_storage(self, source_uri: str, file_type: Optional[str] = None) -> FetchedDocumentBundle:
        """Download from cloud object storage."""
        bucket, key = self.file_manager.parse_uri(source_uri)
        if not key:
            raise ValueError("Object storage URI must include an object key")
        tmp_dir = tempfile.mkdtemp(prefix="document_fetcher_")

        # If key looks like a directory, treat as prefix
        if key.endswith('/'):
             return self._download_prefix(bucket, key, tmp_dir, file_type)

        if self._object_exists(bucket, key):
            local_path = self._safe_join_tmp_dir(
                tmp_dir,
                self._sanitize_filename(os.path.basename(key) or "document"),
            )
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            LOGGER.log_stage(ProcessLogger.STAGE_FETCH, f"Downloading single object: {source_uri}")
            
            stream = self.file_manager.download_file(bucket, key)
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(stream, f)
                
            return FetchedDocumentBundle(document_path=local_path, asset_root=os.path.dirname(local_path), cleanup_root=tmp_dir)
        return self._download_prefix(bucket, key, tmp_dir, file_type)

    def cleanup(self, bundle: Optional[FetchedDocumentBundle]) -> None:
        if not bundle:
            return
        shutil.rmtree(bundle.cleanup_root, ignore_errors=True)

    def _object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists using HEAD request (avoids full download)."""
        try:
            self.file_manager.head_object(bucket, key)
            return True
        except CloudlibException as exc:
            if exc.category == "not_found":
                return False
            raise
        except Exception as e:
            error_str = str(type(e).__name__)
            if "NoSuchKey" in error_str or "404" in str(e) or "Not Found" in str(e):
                return False
            raise

    def _download_prefix(self, bucket: str, prefix: str, tmp_dir: str, file_type: Optional[str] = None) -> FetchedDocumentBundle:
        normalized_prefix = prefix.rstrip("/") + "/"
        
        try:
            keys = self.file_manager.list_files(bucket, prefix=normalized_prefix)
        except Exception:
             # If list fails, maybe it's not a prefix or bucket issue. 
             # But we should probably let it raise or handle it.
             # Original code used paginator which would raise if bucket not found etc.
             raise

        objects: List[str] = []
        candidates: List[str] = []
        LOGGER.log_stage(ProcessLogger.STAGE_FETCH, f"Downloading prefix: {bucket}/{normalized_prefix}")
        
        for key in keys:
            if key.endswith("/"):
                continue
            rel_path = key[len(normalized_prefix) :]
            local_path = self._safe_join_tmp_dir(tmp_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            stream = self.file_manager.download_file(bucket, key)
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(stream, f)
                
            objects.append(local_path)
            if self._is_candidate(local_path, file_type):
                candidates.append(local_path)
                
        if not objects:
            raise FileNotFoundError(f"No objects found under prefix {bucket}/{normalized_prefix}")
        document_path = self._select_primary_document(candidates, file_type)
        LOGGER.log_stage(ProcessLogger.STAGE_FETCH, f"Selected primary document: {document_path}")
        asset_root = os.path.dirname(document_path) if document_path else tmp_dir
        return FetchedDocumentBundle(document_path=document_path, asset_root=asset_root, cleanup_root=tmp_dir)

    def _is_candidate(self, path: str, file_type: Optional[str]) -> bool:
        if file_type:
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            # Handle both 'html' and 'webpage' file types for HTML files
            if file_type.lower() in ('html', 'webpage', 'web_page'):
                return ext in ('html', 'htm')
            return ext == file_type.lower()
        
        # Fallback: accept common document types if no type specified
        return path.lower().endswith(('.html', '.htm', '.pdf', '.docx', '.pptx', '.txt', '.md'))

    def _select_primary_document(self, candidates: List[str], file_type: Optional[str]) -> str:
        if not candidates:
            raise FileNotFoundError(f"No matching files found for type {file_type or 'any'} in the downloaded prefix")

        def priority(path: str) -> tuple[int, str]:
            name = os.path.basename(path).lower()
            if name == "page.html":
                return (0, path)
            if name == "index.html":
                return (1, path)
            return (2, path)

        return sorted(candidates, key=priority)[0]
