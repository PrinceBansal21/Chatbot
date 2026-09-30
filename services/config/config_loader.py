"""Configuration loader service backed by the selected cloud configuration store."""
from __future__ import annotations

import os
from typing import Any, Dict

from cloud_lib.factory import CloudFactory
from cloud_lib import ParameterConfigLoader

from utils.dict_utils import deep_merge
from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


class ConfigLoader:
    """Load required pipeline configuration from canonical configuration paths."""

    APPLICATION_CONFIG_NAMES = (
        "processing",
        "ai-models",
        "vectordb-schema",
        "file-handlers",
    )

    def __init__(self) -> None:
        self.config_manager = CloudFactory.get_config_manager()

    def load(self) -> Dict[str, Any]:
        """
        Load and merge configuration from all sources.
        
        Returns:
            Merged configuration dictionary with normalized keys
        """
        environment = os.environ.get("APP_ENV", "").strip()
        if not environment:
            raise RuntimeError("APP_ENV must be set to load pipeline configuration from the cloud configuration store.")

        application_config: Dict[str, Any] = {}
        for config_name in self.APPLICATION_CONFIG_NAMES:
            parameter_name = f"/win-suite/{environment}/config/{config_name}"
            application_config = self._deep_merge(
                application_config,
                self._load_required_parameter(parameter_name),
            )

        infrastructure_parameter = f"/win-suite/{environment}/infrastructure-config"
        infrastructure_config = self._extract_provider_config(
            self._load_required_parameter(infrastructure_parameter)
        )
        config = self._normalize_application_config(application_config)
        config = self._merge_infrastructure(config, infrastructure_config)
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Configuration loaded from required cloud configuration entries")
        return config
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries. Override values take precedence."""
        return deep_merge(base, override)

    def _load_required_parameter(self, parameter_name: str) -> Dict[str, Any]:
        """Load one required JSON object through the cloud configuration manager."""
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Loading configuration entry: {parameter_name}")
        try:
            config = ParameterConfigLoader(self.config_manager).load(parameter_name)
        except Exception as exc:
            raise RuntimeError(
                f"Required pipeline configuration parameter '{parameter_name}' could not be loaded: {exc}"
            ) from exc
        if not isinstance(config, dict):
            raise RuntimeError(
                f"Required pipeline configuration parameter '{parameter_name}' must contain a JSON object."
            )
        return config

    def _extract_provider_config(self, infra_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the active cloud provider's configuration from nested structure.
        
        Provider can be overridden via CLOUD_PROVIDER environment variable.
        """
        # Determine active provider (env var takes precedence)
        active_provider = os.environ.get("CLOUD_PROVIDER") or infra_config.get("active_provider")
        if not active_provider:
            raise ValueError(
                "Infrastructure config must define active_provider or CLOUD_PROVIDER"
            )
        active_provider = active_provider.lower()
        
        if active_provider not in infra_config:
            available = [k for k in infra_config.keys() if k not in ("active_provider",)]
            raise ValueError(
                f"Cloud provider '{active_provider}' not found in infrastructure config. "
                f"Available providers: {available}"
            )
        
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Using cloud provider: {active_provider}")
        
        # Extract provider-specific config and add cloud_provider key
        provider_config = infra_config[active_provider].copy()
        provider_config["cloud_provider"] = active_provider
        
        return provider_config

    def _normalize_application_config(self, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize application config to the canonical strategy and cloud adapter keys.
        
        Converts nested default_provider/default_strategy patterns to flat structure:
        - embedding.default_provider + embedding.cloud -> embedding with provider + model_id
        - chunking.default_strategy + chunking.recursive -> chunking with strategy + params
        - markdown_conversion.default_provider -> markdown_conversion with provider
        """
        config: Dict[str, Any] = {}
        
        # Keep original structure for reference
        config["markdown_conversion"] = app_config.get("markdown_conversion", {})
        
        # Normalize chunking
        if "chunking" in app_config:
            chunking = app_config["chunking"]
            strategy = chunking.get("default_strategy", "structured")
            strategy_config = chunking.get(strategy, {})
            config["chunking"] = {
                "strategy": strategy,
                **strategy_config,
                # Keep nested strategies for flexibility
                "recursive": chunking.get("recursive", {}),
                "semantic": chunking.get("semantic", {}),
                "contextual": chunking.get("contextual", {}),
                "hierarchical": chunking.get("hierarchical", {}),
            }
        
        # Normalize embedding
        if "embedding" in app_config:
            embedding = app_config["embedding"]
            provider = embedding.get("default_provider", "cloud")
            if provider != "cloud":
                raise ValueError(
                    f"Unsupported embedding provider {provider!r}; expected 'cloud'."
                )
            provider_config = embedding.get(provider, {})
            config["embedding"] = {
                "provider": provider,
                **provider_config,
                "cloud": embedding.get("cloud", {}),
            }
        
        # Normalize vision
        if "vision" in app_config:
            vision = app_config["vision"]
            provider = vision.get("default_provider", "cloud")
            if provider != "cloud":
                raise ValueError(
                    f"Unsupported vision provider {provider!r}; expected 'cloud'."
                )
            provider_config = vision.get(provider, {})
            config["vision"] = {
                "provider": provider,
                **provider_config,
                "cloud": vision.get("cloud", {}),
            }
        
        # PII config (already flat)
        if "pii" in app_config:
            config["pii"] = app_config["pii"]
        
        # Image extraction config (for extracting and storing images from documents)
        if "image_extraction" in app_config:
            config["image_extraction"] = app_config["image_extraction"]
        
        # Image storage config (bucket/prefix for storing extracted images)
        if "image_storage" in app_config:
            config["image_storage"] = app_config["image_storage"]
        
        # Summary generation config (for document summaries)
        if "summary_generation" in app_config:
            config["summary_generation"] = app_config["summary_generation"]
        
        # Chunk validation config (for validating/truncating chunks before embedding)
        if "chunk_validation" in app_config:
            config["chunk_validation"] = app_config["chunk_validation"]
        
        # LLM config (for general LLM usage, prompts, etc.)
        if "llm" in app_config:
            config["llm"] = app_config["llm"]
        
        # Chunk fields schema
        if "chunk_fields" in app_config:
            config["chunk_fields"] = app_config["chunk_fields"]
        
        # File type configs
        if "file_type_configs" in app_config:
            config["file_type_configs"] = app_config["file_type_configs"]

        # AI model configs
        for key in ("model_defaults", "vision_llm_models", "model_token_limits",
                    "inference_profile_required_models", "embedding_supported_dimensions",
                    "config_refresh_ttl_seconds"):
            if key in app_config:
                config[key] = app_config[key]
        
        return config

    def _merge_infrastructure(self, config: Dict[str, Any], infra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge infrastructure config into the canonical application-facing shape.

        Normalizes provider infrastructure keys without exposing provider service names:
        - storage.markdown_output -> markdown_storage
        - storage.encryption -> storage_encryption
        - vectordb -> vectordb
        - deployment -> deployment_dependencies
        - provider compute settings -> worker
        - provider build settings -> build
        - queues -> queues and deployment_dependencies.queues
        """
        # Direct mappings
        if "region" in infra:
            config["region"] = infra["region"]
        
        if "cloud_provider" in infra:
            config["cloud_provider"] = infra["cloud_provider"]
        
        # Database
        if "database" in infra:
            config["database"] = infra["database"]
        
        # Storage normalization
        if "storage" in infra:
            storage = infra["storage"]
            if "markdown_output" in storage:
                config["markdown_storage"] = storage["markdown_output"]
            if "encryption" in storage:
                config["storage_encryption"] = storage["encryption"]

        if "queues" in infra:
            config["queues"] = infra["queues"]

        if "vectordb" in infra:
            config["vectordb"] = infra["vectordb"]

        if "worker" in infra:
            config["worker"] = infra["worker"]

        if "deployment" in infra:
            deployment = infra["deployment"]
            config["deployment_dependencies"] = {
                "container_repository": deployment.get("container_repository"),
                "queues": infra.get("queues", {}),
                "config_parameters": deployment.get("config_parameters", {}),
                "vectordb_index": infra.get("vectordb", {}).get("index"),
            }

            if "worker" in deployment:
                config["worker"] = deployment["worker"]
            if "build" in deployment:
                config["build"] = deployment["build"]

        return config

