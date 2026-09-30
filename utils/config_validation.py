"""
Configuration Validation Utilities

Validates file_type_configs and other configuration structures.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


# Valid strategy/provider values
VALID_MARKDOWN_PROVIDERS = {
    "docling", "vision_llm", "hybrid", "image_aware", "pptx_vision", "parallel_hybrid_pdf",
}

VALID_CHUNKING_STRATEGIES = {
    "semantic", "recursive", "hierarchical", "contextual", "spreadsheet",
}

VALID_EMBEDDING_PROVIDERS = {
    "cloud",
}

VALID_FILE_TYPES = {
    "pdf", "html", "webpage", "docx", "xlsx", "xls", "csv", "pptx", "txt", "image", "email", "default",
}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def validate_file_type_configs(
    config: Dict[str, Any],
    raise_on_error: bool = False,
) -> List[str]:
    """
    Validate file_type_configs section of configuration.
    
    Args:
        config: Full application configuration
        raise_on_error: If True, raise ConfigValidationError on validation failures
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    file_type_configs = config.get("file_type_configs", {})
    if not file_type_configs:
        # No file type configs is valid - will use global defaults
        return errors
    
    if not isinstance(file_type_configs, dict):
        errors.append("file_type_configs must be a dictionary")
        if raise_on_error:
            raise ConfigValidationError(errors)
        return errors
    
    # Track inheritance to detect cycles
    visited_types: Set[str] = set()
    
    for file_type, type_config in file_type_configs.items():
        type_errors = _validate_single_file_type(
            file_type, type_config, file_type_configs, visited_types
        )
        errors.extend(type_errors)
    
    if errors:
        for error in errors:
            LOGGER.log_warning("CONFIG", f"Config validation warning: {error}")
        
        if raise_on_error:
            raise ConfigValidationError(errors)
    
    return errors


def _validate_single_file_type(
    file_type: str,
    type_config: Dict[str, Any],
    all_configs: Dict[str, Any],
    visited: Set[str],
) -> List[str]:
    """Validate a single file type configuration."""
    errors = []
    
    if not isinstance(type_config, dict):
        errors.append(f"file_type_configs.{file_type} must be a dictionary")
        return errors
    
    # Check for unknown file types (warning only)
    if file_type not in VALID_FILE_TYPES and not file_type.startswith("_"):
        LOGGER.log_stage("CONFIG", f"Custom file type configured: {file_type}")
    
    # Validate inheritance
    if "_inherits" in type_config:
        parent = type_config["_inherits"]
        if parent == file_type:
            errors.append(f"file_type_configs.{file_type} cannot inherit from itself")
        elif parent not in all_configs:
            errors.append(
                f"file_type_configs.{file_type}._inherits references unknown type: {parent}"
            )
        elif parent in visited:
            errors.append(
                f"file_type_configs.{file_type} creates circular inheritance"
            )
    
    visited.add(file_type)
    
    # Validate markdown config
    if "markdown" in type_config:
        markdown_errors = _validate_markdown_config(file_type, type_config["markdown"])
        errors.extend(markdown_errors)
    
    # Validate chunking config
    if "chunking" in type_config:
        chunking_errors = _validate_chunking_config(file_type, type_config["chunking"])
        errors.extend(chunking_errors)
    
    # Validate embedding config
    if "embedding" in type_config:
        embedding_errors = _validate_embedding_config(file_type, type_config["embedding"])
        errors.extend(embedding_errors)
    
    # Validate image_extraction config
    if "image_extraction" in type_config:
        image_errors = _validate_image_extraction_config(file_type, type_config["image_extraction"])
        errors.extend(image_errors)
    
    return errors


def _validate_markdown_config(file_type: str, config: Dict[str, Any]) -> List[str]:
    """Validate markdown configuration."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append(f"file_type_configs.{file_type}.markdown must be a dictionary")
        return errors
    
    provider = config.get("provider")
    if provider and provider not in VALID_MARKDOWN_PROVIDERS:
        errors.append(
            f"file_type_configs.{file_type}.markdown.provider '{provider}' is not recognized. "
            f"Valid options: {', '.join(sorted(VALID_MARKDOWN_PROVIDERS))}"
        )
    
    # Validate docling config if present
    if "docling" in config:
        docling_cfg = config["docling"]
        if isinstance(docling_cfg, dict):
            if "ocr" in docling_cfg:
                ocr_cfg = docling_cfg["ocr"]
                if isinstance(ocr_cfg, dict):
                    engine = ocr_cfg.get("engine")
                    valid_engines = {"auto", "easyocr", "rapidocr", "tesseract", "tesseract_cli", "ocrmac"}
                    if engine and engine not in valid_engines:
                        errors.append(
                            f"file_type_configs.{file_type}.markdown.docling.ocr.engine '{engine}' "
                            f"is not valid. Options: {', '.join(sorted(valid_engines))}"
                        )
    
    return errors


def _validate_chunking_config(file_type: str, config: Dict[str, Any]) -> List[str]:
    """Validate chunking configuration."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append(f"file_type_configs.{file_type}.chunking must be a dictionary")
        return errors
    
    strategy = config.get("strategy")
    if strategy and strategy not in VALID_CHUNKING_STRATEGIES:
        errors.append(
            f"file_type_configs.{file_type}.chunking.strategy '{strategy}' is not recognized. "
            f"Valid options: {', '.join(sorted(VALID_CHUNKING_STRATEGIES))}"
        )
    
    # Validate numeric options
    chunk_size = config.get("chunk_size")
    if chunk_size is not None:
        if not isinstance(chunk_size, (int, float)) or chunk_size <= 0:
            errors.append(
                f"file_type_configs.{file_type}.chunking.chunk_size must be a positive number"
            )
    
    overlap = config.get("overlap")
    if overlap is not None:
        if not isinstance(overlap, (int, float)) or overlap < 0:
            errors.append(
                f"file_type_configs.{file_type}.chunking.overlap must be a non-negative number"
            )
    
    return errors


def _validate_embedding_config(file_type: str, config: Dict[str, Any]) -> List[str]:
    """Validate embedding configuration."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append(f"file_type_configs.{file_type}.embedding must be a dictionary")
        return errors
    
    provider = config.get("provider")
    if provider and provider not in VALID_EMBEDDING_PROVIDERS:
        errors.append(
            f"file_type_configs.{file_type}.embedding.provider '{provider}' is not recognized. "
            f"Valid options: {', '.join(sorted(VALID_EMBEDDING_PROVIDERS))}"
        )
    
    return errors


def _validate_image_extraction_config(file_type: str, config: Dict[str, Any]) -> List[str]:
    """Validate image extraction configuration."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append(f"file_type_configs.{file_type}.image_extraction must be a dictionary")
        return errors
    
    # Validate boolean options
    bool_options = [
        "enabled", "store_images", "generate_descriptions", "embed_in_markdown",
        "create_image_chunks", "link_images_to_chunks", "skip_decorative",
        "fetch_remote", "convert_svg_to_png",
    ]
    
    for opt in bool_options:
        if opt in config and not isinstance(config[opt], bool):
            errors.append(
                f"file_type_configs.{file_type}.image_extraction.{opt} must be a boolean"
            )
    
    # Validate numeric options
    num_options = ["min_width", "min_height", "max_images_per_chunk"]
    for opt in num_options:
        if opt in config:
            val = config[opt]
            if not isinstance(val, (int, float)) or val < 0:
                errors.append(
                    f"file_type_configs.{file_type}.image_extraction.{opt} must be a non-negative number"
                )
    
    return errors

