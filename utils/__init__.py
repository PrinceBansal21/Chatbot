"""Utility functions for the processor package.

This module contains utilities - stateless helper functions and lightweight classes.
Core services should be imported directly from the services package.
"""

# Logging utilities
from .logger import (
    ProcessLogger,
    get_logger,
    set_file_id,
    clear_file_id,
    get_file_id,
    configure_root_logger,
    reset_logging,
)

# Payload and metadata utilities
from .payload_utils import (
    extract_base_metadata,
    get_source_uri,
    resolve_location_label,
    PayloadParser,
)

# Result building utilities
from .result_builder import ResultBuilder

# Image-related utilities
from .image_utils import (
    ImageConfigResolver,
    MarkdownPreprocessor,
    MetadataEnricher,
    ImageChunkLinker,
    ImageRelevanceScorer,
    link_images_to_chunks,
)

# Dictionary utilities
from .dict_utils import deep_merge

# Config validation
from .config_validation import (
    validate_file_type_configs,
    ConfigValidationError,
)

__all__ = [
    # Logging utilities
    "ProcessLogger",
    "get_logger",
    "set_file_id",
    "clear_file_id",
    "get_file_id",
    "configure_root_logger",
    "reset_logging",
    # Payload utilities
    "extract_base_metadata",
    "resolve_location_label",
    "get_source_uri",
    "PayloadParser",
    # Result utilities
    "ResultBuilder",
    # Image utilities
    "ImageConfigResolver",
    "MarkdownPreprocessor",
    "MetadataEnricher",
    "ImageChunkLinker",
    "link_images_to_chunks",
    # Dictionary utilities
    "deep_merge",
    # Config validation
    "validate_file_type_configs",
    "ConfigValidationError",
]

