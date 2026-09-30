"""Markdown processing services (conversion and persistence)."""
from __future__ import annotations

import logging
import posixpath
import unicodedata
from io import BytesIO
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from cloud_lib.factory import CloudFactory
from rag_lib.factory import RAGFactory

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


def sanitize_metadata_for_storage(metadata: Dict[str, str], max_total_size: int = 1800) -> Dict[str, str]:
    """
    Sanitize metadata values for object storage compatibility.
    
    Object storage metadata constraints:
    - Can only contain ASCII characters
    - Total user-defined metadata limited to 2KB (we use 1800 bytes for safety margin)
    
    This function:
    1. Excludes fields that don't need to be in object storage metadata (e.g., page_summary)
    2. Converts non-ASCII characters to ASCII equivalents
    3. Truncates individual values that are too long
    4. Ensures total metadata size stays within limits
    
    Note: page_summary is excluded from object storage metadata because:
    - It is only used for vector retrieval (stored in chunk metadata)
    - It often contains non-ASCII characters from LLM output
    - Object storage metadata has strict ASCII and size limits
    
    Args:
        metadata: Dictionary of metadata key-value pairs
        max_total_size: Maximum total size in bytes for all metadata (default 1800)
        
    Returns:
        Dictionary with ASCII-safe and size-compliant metadata values
    """
    # Keys to exclude from object storage metadata (stored elsewhere or not needed there)
    EXCLUDED_KEYS = {'page_summary', 'document_summary', 'summary'}
    
    # Maximum length for any single metadata value (leave room for key names)
    MAX_VALUE_LENGTH = 500
    
    # Common character replacements for ASCII compatibility
    replacements = {
        '®': '(R)',
        '™': '(TM)',
        '©': '(C)',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '°': ' degrees',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '•': '*',
        '·': '*',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
    }
    
    def sanitize_value(value: str, max_length: int = MAX_VALUE_LENGTH) -> str:
        if not isinstance(value, str):
            value = str(value)
        
        # Apply common replacements first
        for char, replacement in replacements.items():
            value = value.replace(char, replacement)
        
        # Normalize unicode (decompose accented characters)
        value = unicodedata.normalize('NFKD', value)
        
        # Keep only ASCII characters
        value = value.encode('ascii', 'ignore').decode('ascii')
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length - 3] + '...'
        
        return value
    
    # First pass: filter out excluded keys and sanitize remaining values
    sanitized = {
        key: sanitize_value(value) 
        for key, value in metadata.items() 
        if key not in EXCLUDED_KEYS
    }
    
    # Second pass: ensure total size is within limits
    # Calculate total size (key + value for each entry)
    def calculate_total_size(meta: Dict[str, str]) -> int:
        return sum(len(k) + len(v) for k, v in meta.items())
    
    total_size = calculate_total_size(sanitized)
    
    # If still too large, progressively truncate the longest values
    while total_size > max_total_size and sanitized:
        # Find the longest value
        longest_key = max(sanitized.keys(), key=lambda k: len(sanitized[k]))
        current_len = len(sanitized[longest_key])
        
        if current_len <= 50:
            # Values are already very short, remove the longest one entirely
            LOGGER.log_warning(
                ProcessLogger.STAGE_STORAGE,
                f"Removing metadata key '{longest_key}' to fit object storage size limit"
            )
            del sanitized[longest_key]
        else:
            # Truncate the longest value by half
            new_len = max(50, current_len // 2)
            sanitized[longest_key] = sanitized[longest_key][:new_len - 3] + '...'
        
        total_size = calculate_total_size(sanitized)
    
    return sanitized


class MarkdownConverter:
    """Converts supported documents into Markdown."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, default_provider: str = "parallel_hybrid_pdf"):
        """
        Initialize the markdown converter.
        
        Args:
            config: Full markdown_conversion config dict with provider-specific settings
            default_provider: Default provider to use (can be overridden in config)
        """
        self.config = config or {}
        self.default_provider = self.config.get("default_provider", default_provider)
        # Cache managers to avoid reloading - keyed by (provider, config_hash)
        self._managers = {}

    def _get_config_hash(self, provider_config: Dict[str, Any]) -> str:
        """Generate a hash of provider config for caching."""
        import json
        import hashlib
        config_str = json.dumps(provider_config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def _get_manager(self, provider: str, provider_config: Optional[Dict[str, Any]] = None):
        """
        Get or create a markdown manager for the given provider.
        
        Args:
            provider: Provider name (e.g., 'docling', 'hybrid')
            provider_config: Optional provider-specific configuration overrides
        """
        # Use provided config or fall back to stored config
        if provider_config is None:
            provider_config = self.config.get(provider, {})
        
        # Create cache key from provider and config
        config_hash = self._get_config_hash(provider_config)
        cache_key = f"{provider}_{config_hash}"
        
        if cache_key not in self._managers:
            self._managers[cache_key] = RAGFactory.get_markdown_manager(provider, **provider_config)
        
        return self._managers[cache_key]

    def convert(
        self, 
        file_path: str, 
        mime_type: Optional[str] = None, 
        provider: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        **convert_kwargs,
    ) -> str:
        """
        Convert a document to markdown.
        
        Args:
            file_path: Path to the document file
            mime_type: Optional MIME type hint
            provider: Markdown conversion provider to use
            provider_config: Optional provider-specific configuration (for file-type configs)
            **convert_kwargs: Additional kwargs forwarded to the adapter's convert()
            
        Returns:
            Markdown string
        """
        provider = provider or self.default_provider
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT,
            f"Converting document using provider '{provider}'",
            extra_data={
                "file_path": file_path,
                "mime_type": mime_type,
                "has_provider_config": provider_config is not None,
            },
        )
        
        import time
        start = time.time()
        manager = self._get_manager(provider, provider_config)
        result = manager.convert(file_path=file_path, mime_type=mime_type, **convert_kwargs)
        elapsed = time.time() - start

        # Capture auto-detected infographic pages (if the adapter exposes them)
        self.detected_infographic_pages = getattr(
            manager, "detected_infographic_pages", []
        )
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT,
            f"Markdown conversion complete: {len(result)} chars in {elapsed:.1f}s",
            extra_data={
                "provider": provider,
                "content_length": len(result),
                "duration_seconds": round(elapsed, 2),
            },
        )
        
        return result


class MarkdownRepository:
    """Persists Markdown outputs back into Storage."""

    def __init__(self, region: str, bucket_uri: str, encryption_key: Optional[str]):
        self.file_manager = CloudFactory.get_file_manager(bucket_key="markdown_output")
        self.bucket, self.base_prefix = self.file_manager.parse_uri(bucket_uri)
        self.uri_scheme = urlsplit(bucket_uri).scheme
        self.encryption_key = encryption_key

    def put(self, file_id: str, markdown: str, metadata: Dict[str, str],
            document_prefix: Optional[str] = None) -> str:
        """Persist markdown to cloud storage.

        Args:
            file_id: Unique file identifier.
            markdown: Markdown content to store.
            metadata: Metadata dict to attach as object storage metadata.
            document_prefix: Optional object key prefix where the
                original document lives.  When provided the markdown is stored
                alongside the document as ``{document_prefix}/content.md``
                instead of the legacy ``{file_id}/content.md`` flat layout.
        """
        if document_prefix:
            key = posixpath.join(document_prefix, "content.md").strip("/")
        else:
            key = posixpath.join(self.base_prefix, file_id, "content.md").strip("/")
        
        # Sanitize metadata for object storage (allows only ASCII characters)
        safe_metadata = sanitize_metadata_for_storage(metadata)
        
        extra_args = {"Metadata": safe_metadata, "ContentType": "text/markdown"}
        extra_args.update(self.file_manager.encryption_kwargs(self.encryption_key))
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Storing markdown to {self.uri_scheme}://{self.bucket}/{key} ({len(markdown)} chars)",
        )
        
        file_obj = BytesIO(markdown.encode("utf-8"))
        self.file_manager.upload_file(file_obj, self.bucket, key, **extra_args)
        
        location = f"{self.uri_scheme}://{self.bucket}/{key}"
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Markdown stored successfully: {location}",
        )
        
        return location
