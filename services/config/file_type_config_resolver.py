"""
File Type Configuration Resolver

Resolves processing configuration based on file type with support for:
- File-type specific configurations
- Inheritance between similar file types
- Payload overrides
- Fallback to global defaults
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

from utils.dict_utils import deep_merge
from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


# File extension to canonical file type mapping
# (cloud-agnostic; used by all document processors regardless of backend)
EXTENSION_TO_FILE_TYPE: Dict[str, str] = {
    # PDF
    ".pdf": "pdf",
    # HTML/Web
    ".html": "html",
    ".htm": "html",
    # Word Documents
    ".docx": "docx",
    ".doc": "docx",
    # Excel
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "xlsx",
    # PowerPoint
    ".pptx": "pptx",
    ".ppt": "pptx",
    # Text
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    # Images
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    # Email
    ".eml": "email",
    ".msg": "email",
}

# File types that map to "webpage" (URLs, not local HTML files)
WEBPAGE_INDICATORS = {"webpage", "web_page", "weburl", "url"}

# Generic file types that should fall back to extension detection
# These don't have specific configs and need to be resolved from the actual file extension
GENERIC_FILE_TYPES = {"document", "file", "unknown", "binary", "attachment"}


class FileTypeConfigResolver:
    """
    Resolves processing configuration based on file type.
    
    Configuration priority (highest to lowest):
    1. Payload overrides (explicit per-request settings)
    2. File-type specific configuration
    3. Inherited configuration (via _inherits key)
    4. Global defaults
    
    Example config structure:
    {
        "file_type_configs": {
            "pdf": {
                "markdown": {"provider": "docling", "ocr": {"enabled": true}},
                "image_extraction": {"enabled": true},
                "chunking": {"strategy": "recursive"}
            },
            "html": {
                "markdown": {"provider": "docling"},
                "image_extraction": {"enabled": true, "skip_decorative": true},
                "chunking": {"strategy": "semantic"}
            },
            "webpage": {
                "_inherits": "html",
                "image_extraction": {"fetch_remote": true}
            },
            "default": {
                "markdown": {"provider": "docling"},
                "image_extraction": {"enabled": false},
                "chunking": {"strategy": "recursive"}
            }
        }
    }
    """
    
    # Sections that can be configured per file type
    CONFIG_SECTIONS = [
        "markdown",
        "image_extraction", 
        "chunking",
        "embedding",
        "vision",
        "pii",
        "summary_generation",
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FileTypeConfigResolver.
        
        Args:
            config: Full application configuration dictionary
        """
        self.config = config
        self.file_type_configs = config.get("file_type_configs", {})
        self._resolved_cache: Dict[str, Dict[str, Any]] = {}
    
    def resolve(
        self,
        file_type: Optional[str] = None,
        file_path: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve complete processing configuration for a file type.
        
        Args:
            file_type: Explicit file type (e.g., 'pdf', 'html', 'webpage')
            file_path: File path to detect type from extension
            payload: Request payload with potential overrides
            
        Returns:
            Merged configuration dictionary with all processing settings
        """
        payload = payload or {}
        
        # Determine canonical file type
        canonical_type = self._resolve_file_type(file_type, file_path, payload)
        LOGGER.log_stage(
            ProcessLogger.STAGE_INPUT,
            f"Resolved file type: {canonical_type}"
        )
        
        # Get base configuration for this file type (with caching)
        if canonical_type not in self._resolved_cache:
            self._resolved_cache[canonical_type] = self._build_type_config(canonical_type)
        
        base_config = copy.deepcopy(self._resolved_cache[canonical_type])
        
        # Apply payload overrides
        resolved = self._apply_payload_overrides(base_config, payload)
        
        # Add resolved file type to config
        resolved["_resolved_file_type"] = canonical_type
        
        return resolved
    
    def _resolve_file_type(
        self,
        explicit_type: Optional[str],
        file_path: Optional[str],
        payload: Dict[str, Any],
    ) -> str:
        """
        Determine the canonical file type.
        
        Priority:
        1. Explicit file_type parameter (unless it's generic like "document")
        2. file_type in payload (unless it's generic like "document")
        3. Extension from file_path
        4. Extension from source_uri in payload
        5. Fall back to generic type if provided
        6. Default
        
        Generic types like "document", "file", "unknown" trigger extension detection
        because they don't have specific processing configs.
        """
        generic_type_provided = None
        
        # Check explicit parameter
        if explicit_type:
            normalized = self._normalize_file_type(explicit_type)
            if normalized not in GENERIC_FILE_TYPES:
                return normalized
            generic_type_provided = normalized
        
        # Check payload file_type
        payload_type = payload.get("file_type")
        if payload_type:
            normalized = self._normalize_file_type(payload_type)
            if normalized not in GENERIC_FILE_TYPES:
                return normalized
            generic_type_provided = generic_type_provided or normalized
        
        # For generic types, try to detect from file path or source_uri
        # Check file_path
        if file_path:
            detected = self._detect_from_path(file_path)
            if detected:
                return detected
        
        # Check source_uri in payload (including web_url)
        source_uri = (
            payload.get("source_uri") or
            payload.get("storage_location") or
            payload.get("web_url")
        )
        if source_uri:
            detected = self._detect_from_path(source_uri)
            if detected:
                return detected
            # HTTP/HTTPS URL without a recognizable file extension → treat as webpage
            if source_uri.lower().startswith(("http://", "https://")):
                return "webpage"
        
        # If we had a generic type, return it as fallback (better than "default")
        if generic_type_provided:
            return generic_type_provided
        
        return "default"
    
    # Dotless aliases: payload file_type values (e.g. "ppt") that should
    # map to their canonical config key (e.g. "pptx").
    _TYPE_ALIASES: dict = {
        "ppt": "pptx",
        "doc": "docx",
        "xls": "xlsx",
        "xlsm": "xlsx",
        "htm": "html",
    }

    def _normalize_file_type(self, file_type: str) -> str:
        """Normalize file type string to canonical form."""
        file_type = file_type.lower().strip()
        
        # Handle webpage indicators
        if file_type in WEBPAGE_INDICATORS:
            return "webpage"
        
        # Handle extension format (e.g., ".pdf" -> "pdf")
        if file_type.startswith("."):
            return EXTENSION_TO_FILE_TYPE.get(file_type, file_type[1:])
        
        # Handle dotless aliases (e.g., "ppt" -> "pptx")
        return self._TYPE_ALIASES.get(file_type, file_type)
    
    def _detect_from_path(self, path: str) -> Optional[str]:
        """Detect file type from file path or URI."""
        # URI schemes do not affect extension-based detection.
        path = path.split("?")[0]  # Remove query params
        
        # Get extension
        _, ext = os.path.splitext(path.lower())
        if ext:
            return EXTENSION_TO_FILE_TYPE.get(ext)
        
        return None
    
    def _build_type_config(self, file_type: str) -> Dict[str, Any]:
        """
        Build complete configuration for a file type.
        
        Handles inheritance and merges with global defaults.
        """
        # Get file-type specific config
        type_config = self.file_type_configs.get(file_type, {})
        
        # Handle inheritance
        inherited_config = {}
        if "_inherits" in type_config:
            parent_type = type_config["_inherits"]
            if parent_type != file_type:  # Prevent circular inheritance
                inherited_config = self._build_type_config(parent_type)
        
        # Get default file type config (supports both "default" and "_default" keys)
        default_config = self.file_type_configs.get("default") or self.file_type_configs.get("_default", {})
        
        # Build merged config for each section
        result = {}
        for section in self.CONFIG_SECTIONS:
            result[section] = self._merge_section(
                section,
                global_config=self._get_global_section(section),
                default_type_config=default_config.get(section, {}),
                inherited_config=inherited_config.get(section, {}),
                type_config=type_config.get(section, {}),
            )
        
        return result
    
    def _get_global_section(self, section: str) -> Dict[str, Any]:
        """Get global configuration for a section."""
        # Map section names to config keys
        section_mapping = {
            "markdown": "markdown_conversion",
            "image_extraction": "image_extraction",
            "chunking": "chunking",
            "embedding": "embedding",
            "vision": "vision",
            "pii": "pii",
            "summary_generation": "summary_generation",
        }
        config_key = section_mapping.get(section, section)
        return self.config.get(config_key, {})
    
    def _merge_section(
        self,
        section: str,
        global_config: Dict[str, Any],
        default_type_config: Dict[str, Any],
        inherited_config: Dict[str, Any],
        type_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deep merge configurations for a section.
        
        Merge order (later overrides earlier):
        1. Global config
        2. Default file type config
        3. Inherited config
        4. Specific file type config
        """
        result = copy.deepcopy(global_config)
        
        for cfg in [default_type_config, inherited_config, type_config]:
            result = self._deep_merge(result, cfg)
        
        return result
    
    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries. Override values take precedence."""
        return deep_merge(base, override, deep_copy=True, skip_private_keys=True)
    
    def _apply_payload_overrides(
        self,
        base_config: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply payload-level overrides to the configuration.
        
        Supports both flat keys and nested structures:
        - {"chunking_strategy": "semantic"} -> chunking.strategy
        - {"chunking": {"strategy": "semantic"}} -> chunking.strategy
        - {"image_extraction": {"enabled": false}} -> image_extraction.enabled
        """
        result = copy.deepcopy(base_config)
        
        # Handle flat key overrides
        flat_mappings = {
            "markdown_provider": ("markdown", "provider"),
            "chunking_strategy": ("chunking", "strategy"),
            "embedding_provider": ("embedding", "provider"),
            "vision_provider": ("vision", "provider"),
        }
        
        for payload_key, (section, config_key) in flat_mappings.items():
            if payload_key in payload and payload[payload_key]:
                if section not in result:
                    result[section] = {}
                result[section][config_key] = payload[payload_key]
        
        # Handle nested overrides
        for section in self.CONFIG_SECTIONS:
            if section in payload and isinstance(payload[section], dict):
                if section not in result:
                    result[section] = {}
                result[section] = self._deep_merge(result[section], payload[section])
        
        return result
    
    def get_markdown_config(self, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract markdown conversion configuration."""
        return resolved_config.get("markdown", {})
    
    def get_image_extraction_config(self, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract image extraction configuration."""
        return resolved_config.get("image_extraction", {})
    
    def get_chunking_config(self, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chunking configuration."""
        return resolved_config.get("chunking", {})
    

