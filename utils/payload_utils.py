"""Payload parsing, normalization, and metadata utilities for the ingestion pipeline."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from .logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)

# Priority-ordered field names for source URI resolution
_SOURCE_URI_FIELDS = ("source_uri", "storage_location", "web_url")


def get_source_uri(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract the source URI from a payload, checking multiple possible fields.

    The fields are checked in priority order:
    ``source_uri`` > ``storage_location`` > ``web_url``.

    Returns:
        The first non-empty value found, or *None*.
    """
    for field in _SOURCE_URI_FIELDS:
        val = payload.get(field)
        if val:
            return val
    return None


# File types whose page_number actually represents a slide index.
_SLIDE_FILE_TYPES = ("ppt", "pptx")


def resolve_location_label(resolved_file_type: Optional[str]) -> str:
    """Return the user-facing location label for a resolved file type.

    Downstream consumers (chat backend, frontend) render ``{label} {page_number}``
    for citations and image references. PPT/PPTX documents page through
    slides, so they get "Slide" instead of the default "Page".
    """
    ft = (resolved_file_type or "").lower().lstrip(".")
    return "Slide" if ft in _SLIDE_FILE_TYPES else "Page"


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def extract_base_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract base metadata from a processing payload.
    
    Normalizes various payload field names and ensures required metadata fields
    are present for downstream processing.
    
    Args:
        payload: Document processing payload containing file_id, source_uri, 
                 file_type, and optional metadata fields
                 
    Returns:
        Dictionary of metadata with normalized field names
    """
    metadata: Dict[str, str] = {
        "file_id": payload["file_id"],
        "source_uri": get_source_uri(payload) or "",
        "file_type": payload.get("file_type", "unknown"),
    }
    
    # Merge any extra metadata from payload
    if extra := payload.get("metadata"):
        metadata.update({str(k): str(v) for k, v in extra.items()})

    # Ensure file_name is present
    if "file_name" not in metadata and "name" in metadata:
        metadata["file_name"] = metadata["name"]
    if "file_name" not in metadata and metadata.get("source_uri"):
        metadata["file_name"] = metadata["source_uri"].rsplit("/", 1)[-1]

    # Ensure web_url is present (normalize different field names)
    if "web_url" not in metadata:
        if "webUrl" in metadata:
            metadata["web_url"] = metadata["webUrl"]
        elif "url" in metadata:
            metadata["web_url"] = metadata["url"]
        elif payload.get("web_url"):
            metadata["web_url"] = payload["web_url"]
        elif payload.get("file_url"):
            metadata["web_url"] = payload["file_url"]

    return metadata


# =============================================================================
# PAYLOAD PARSER
# =============================================================================

class PayloadParser:
    """
    Parses and normalizes incoming request payloads for document processing.
    
    Handles:
    - Auto-generating file_id if not provided
    - Normalizing source URIs from various payload fields
    - Parsing target indices (single, list, or JSON array)
    """
    
    def __init__(self, default_index: str):
        """
        Initialize the payload parser.
        
        Args:
            default_index: Default index name to use when not specified in payload
        """
        self.default_index = default_index
    
    def normalize_payload(self, payload: Dict, operation: str = "index") -> Dict:
        """
        Normalize the payload by auto-generating missing fields.
        
        Args:
            payload: Original request payload
            operation: Operation type ('index' or 'delete')
            
        Returns:
            Normalized payload with auto-generated fields
        """
        # Auto-generate file_id if not provided (e.g. minimal web_url-only payloads)
        file_id = payload.get("file_id")
        if not file_id and operation != "delete":
            file_id = str(uuid.uuid4())
            payload["file_id"] = file_id
            LOGGER.log_stage(
                ProcessLogger.STAGE_INPUT,
                f"Auto-generated file_id: {file_id}"
            )
        
        # Normalize web_url into source_uri if no other source is provided
        if not any(payload.get(k) for k in ("source_uri", "storage_location")):
            web_url = payload.get("web_url")
            if web_url:
                payload["source_uri"] = web_url
                LOGGER.log_stage(
                    ProcessLogger.STAGE_INPUT,
                    f"Using web_url as source_uri: {web_url}"
                )
        
        return payload
    
    def parse_target_indices(self, payload: Dict) -> List[str]:
        """
        Parse index_name from payload - supports string, list, or JSON array.
        
        Args:
            payload: Request payload containing index_name
            
        Returns:
            List of target index names
        """
        index_name = payload.get("index_name")
        
        if index_name is None:
            return [self.default_index]
        
        # Already a list
        if isinstance(index_name, list):
            indices = [str(idx).strip().lower() for idx in index_name if idx and str(idx).strip()]
            return indices if indices else [self.default_index]
        
        # String - could be JSON array or single index
        if isinstance(index_name, str):
            index_name = index_name.strip()
            if not index_name:
                return [self.default_index]
            
            # Try parsing as JSON array
            if index_name.startswith('['):
                try:
                    parsed = json.loads(index_name)
                    if isinstance(parsed, list):
                        indices = [str(idx).strip().lower() for idx in parsed if idx and str(idx).strip()]
                        return indices if indices else [self.default_index]
                except json.JSONDecodeError:
                    pass
            
            # Single index name
            return [index_name.lower()]
        
        return [self.default_index]
    
    def get_source_uri(self, payload: Dict) -> Optional[str]:
        """
        Extract source URI from payload, checking multiple possible fields.
        
        Args:
            payload: Request payload
            
        Returns:
            Source URI or None if not found
        """
        return get_source_uri(payload)
    
    def get_file_id(self, payload: Dict) -> Optional[str]:
        """Get file_id from payload."""
        return payload.get("file_id")
    
    def get_operation(self, payload: Dict) -> str:
        """Get operation type from payload, defaulting to 'index'."""
        return payload.get("operation", "index").lower()


__all__ = [
    "extract_base_metadata",
    "PayloadParser",
]
