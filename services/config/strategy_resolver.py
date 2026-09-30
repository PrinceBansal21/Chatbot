"""
Strategy Resolution Utilities

Handles resolution of processing strategies from payload, database, and config.
Integrates with FileTypeConfigResolver for file-type-specific configurations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import ProcessLogger, get_logger
from .file_type_config_resolver import FileTypeConfigResolver

LOGGER = get_logger(__name__)


class StrategyResolver:
    """
    Resolves processing strategies from multiple sources.
    
    Resolution priority: Payload > Database > File Type Config > Global Default
    
    Now integrates with FileTypeConfigResolver for file-type-specific
    configurations, allowing different processing strategies per document type.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        ingestion_repo=None,
    ):
        """
        Initialize the StrategyResolver.
        
        Args:
            config: Application configuration dictionary
            ingestion_repo: Optional IngestionRepository for DB lookups
        """
        self.config = config
        self.ingestion_repo = ingestion_repo
        self.file_type_resolver = FileTypeConfigResolver(config)
    
    def resolve_all(
        self,
        payload: Dict[str, Any],
        file_id: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Resolve all processing strategies for a file.
        
        Args:
            payload: Request payload
            file_id: File identifier for DB lookup
            file_path: Optional file path for type detection
            
        Returns:
            Dictionary with resolved strategies
        """
        # Fetch DB record if available
        db_record = self._get_db_record(file_id)
        
        # Get file-type-specific configuration
        file_type_config = self.file_type_resolver.resolve(
            file_type=payload.get("file_type"),
            file_path=file_path,
            payload=payload,
        )
        
        # Resolve each strategy with file type awareness
        strategies = {
            "markdown_provider": self._resolve_with_file_type(
                payload, db_record, file_type_config,
                payload_key="markdown_provider",
                db_key="markdown_strategy",
                file_type_section="markdown",
                file_type_key="provider",
                config_section="markdown_conversion",
                config_key="default_provider"
            ),
            "chunking_strategy": self._resolve_with_file_type(
                payload, db_record, file_type_config,
                payload_key="chunking_strategy",
                db_key="chunking_strategy",
                file_type_section="chunking",
                file_type_key="strategy",
                config_section="chunking",
                config_key="default_strategy"
            ),
            "embedding_provider": self._resolve_with_file_type(
                payload, db_record, file_type_config,
                payload_key="embedding_provider",
                db_key="embedding_strategy",
                file_type_section="embedding",
                file_type_key="provider",
                config_section="embedding",
                config_key="default_provider"
            ),
            "vision_provider": self._resolve_with_file_type(
                payload, db_record, file_type_config,
                payload_key="vision_provider",
                db_key="image_description_strategy",
                file_type_section="vision",
                file_type_key="provider",
                config_section="vision",
                config_key="default_provider"
            ),
            # Vector dimension override from payload (None means use infra config default)
            "vector_dimension": self._resolve_vector_dimension(payload),
            # Store resolved file type for reference
            "_resolved_file_type": file_type_config.get("_resolved_file_type", "default"),
        }
        
        # Validate vector_dimension is supported by the resolved embedding provider
        if strategies.get("vector_dimension") is not None:
            self._validate_vector_dimension(
                strategies["embedding_provider"],
                strategies["vector_dimension"],
            )
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_INPUT,
            "Resolved processing strategies",
            extra_data=strategies
        )
        
        return strategies
    
    def resolve_full_config(
        self,
        payload: Dict[str, Any],
        file_id: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve complete processing configuration including all options.
        
        This returns the full merged configuration, not just strategy names.
        Useful when components need detailed configuration (e.g., OCR settings).
        
        Args:
            payload: Request payload
            file_id: File identifier for DB lookup
            file_path: Optional file path for type detection
            
        Returns:
            Complete merged configuration dictionary
        """
        return self.file_type_resolver.resolve(
            file_type=payload.get("file_type"),
            file_path=file_path,
            payload=payload,
        )
    
    def get_markdown_config(
        self,
        payload: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get markdown conversion configuration for file type."""
        full_config = self.file_type_resolver.resolve(
            file_type=payload.get("file_type"),
            file_path=file_path,
            payload=payload,
        )
        return self.file_type_resolver.get_markdown_config(full_config)
    
    def get_image_extraction_config(
        self,
        payload: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get image extraction configuration for file type."""
        full_config = self.file_type_resolver.resolve(
            file_type=payload.get("file_type"),
            file_path=file_path,
            payload=payload,
        )
        return self.file_type_resolver.get_image_extraction_config(full_config)
    
    def get_chunking_config(
        self,
        payload: Dict[str, Any],
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get chunking configuration for file type."""
        full_config = self.file_type_resolver.resolve(
            file_type=payload.get("file_type"),
            file_path=file_path,
            payload=payload,
        )
        return self.file_type_resolver.get_chunking_config(full_config)
    
    def _get_db_record(self, file_id: str) -> Dict[str, Any]:
        """Fetch DB record for strategy resolution."""
        if not self.ingestion_repo:
            return {}
        
        try:
            return self.ingestion_repo.get_status(file_id) or {}
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_DB_UPDATE,
                f"Failed to fetch DB record: {e}"
            )
            return {}
    
    def _resolve_vector_dimension(self, payload: Dict[str, Any]) -> Optional[int]:
        """Return vector_dimension from the payload, or None to use config default.

        Validates that the value is a positive integer when present.
        """
        raw = payload.get("vector_dimension")
        if raw is None:
            return None
        try:
            dim = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"vector_dimension must be a positive integer, got: {raw!r}"
            )
        if dim <= 0:
            raise ValueError(
                f"vector_dimension must be a positive integer, got: {dim}"
            )
        return dim

    def _validate_vector_dimension(
        self,
        embedding_provider: Optional[str],
        vector_dimension: int,
    ) -> None:
        """Validate that vector_dimension is supported by the embedding model.

        Looks up the resolved model ID for *embedding_provider* and checks
        the requested dimension against ``embedding_supported_dimensions``
        in the ai-models cloud configuration entry. Raises ``ValueError`` early — before
        any document processing — if the dimension is unsupported.
        """
        from rag_lib.model_config import (
            get_model_default,
            get_embedding_supported_dimensions,
        )

        if not embedding_provider:
            return

        if embedding_provider != "cloud":
            raise ValueError(
                f"Unsupported embedding provider {embedding_provider!r}; expected 'cloud'."
            )

        from rag_lib.factory import RAGFactory

        config_key = RAGFactory.get_cloud_provider()

        # Check for provider-specific model_id override in embedding config
        embedding_cfg = self.config.get("embedding", {})
        model_id = (
            embedding_cfg.get(config_key, {}).get("model_id")
            or get_model_default("embedding", config_key, fallback=None)
        )

        if not model_id:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INPUT,
                f"Cannot validate vector_dimension={vector_dimension}: "
                f"no model_id resolved for provider '{embedding_provider}'",
            )
            return

        from cloud_lib.model_config import get_base_model_id

        bare_model_id = get_base_model_id(model_id)

        supported_dims = get_embedding_supported_dimensions()
        allowed = supported_dims.get(model_id) or supported_dims.get(bare_model_id)

        if allowed is None:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INPUT,
                f"No supported dimensions listed for model '{model_id}'; "
                f"skipping dimension validation for vector_dimension={vector_dimension}",
            )
            return

        if vector_dimension not in allowed:
            raise ValueError(
                f"vector_dimension={vector_dimension} is not supported by "
                f"embedding model '{model_id}'. "
                f"Supported dimensions: {sorted(allowed)}"
            )

        LOGGER.log_stage(
            ProcessLogger.STAGE_INPUT,
            f"Validated vector_dimension={vector_dimension} for model '{model_id}'",
        )

    def _resolve_with_file_type(
        self,
        payload: Dict[str, Any],
        db_record: Dict[str, Any],
        file_type_config: Dict[str, Any],
        payload_key: str,
        db_key: str,
        file_type_section: str,
        file_type_key: str,
        config_section: str,
        config_key: str = "default_provider",
    ) -> Optional[str]:
        """
        Resolve a single strategy value with file type awareness.
        
        Priority: Payload > File Type Config > DB > Global Default
        """
        # Try payload first
        val = payload.get(payload_key)
        if val:
            return val
        
        # Try file type specific config (takes precedence over DB)
        section_config = file_type_config.get(file_type_section, {})
        val = section_config.get(file_type_key) or section_config.get("default_" + file_type_key)
        if val:
            return val
        
        # Try DB record (fallback if no file type config)
        val = db_record.get(db_key)
        if val:
            return val
        
        # Fall back to global config default
        return self.config.get(config_section, {}).get(config_key)
    
    def update_db_strategies(
        self,
        file_id: str,
        strategies: Dict[str, str],
        index_name: str,
    ) -> None:
        """
        Update database with resolved strategies.
        
        Args:
            file_id: File identifier
            strategies: Resolved strategies dictionary
            index_name: Target index name
        """
        if not self.ingestion_repo:
            return
        
        strategies_update = {
            "markdown_strategy": strategies.get("markdown_provider"),
            "chunking_strategy": strategies.get("chunking_strategy"),
            "embedding_strategy": strategies.get("embedding_provider"),
            "image_description_strategy": strategies.get("vision_provider"),
        }
        
        # Include resolved file type if available
        if "_resolved_file_type" in strategies:
            strategies_update["file_type"] = strategies["_resolved_file_type"]
        
        self.ingestion_repo.mark_processing(
            file_id,
            strategies=strategies_update,
            index_name=index_name
        )
        LOGGER.log_db_update("PROCESSING")
