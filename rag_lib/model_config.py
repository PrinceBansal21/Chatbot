"""Centralised model configuration loader backed by the cloud config store.

Loads model IDs, token limits, and other AI-model metadata from the
canonical ``/win-suite/{APP_ENV}/config/ai-models`` parameter.

Usage::

    from rag_lib.model_config import get_ai_models_config

    cfg = get_ai_models_config()
    embedding_default = cfg["model_defaults"]["embedding"][provider]

The JSON is loaded **once** and cached for the lifetime of the process.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


class ModelConfigError(Exception):
    """Raised when required model configuration is missing or invalid.

    Use this instead of falling back to silent defaults — production
    deployments must fail fast on misconfiguration rather than silently
    invoking the wrong model in the wrong region.
    """


def resolve_config_provider_key(provider: str) -> str:
    """Resolve the cloud adapter key to an explicit model-config namespace."""
    if provider == "cloud":
        provider = os.environ.get("CLOUD_PROVIDER", "").strip().lower()
    if not provider:
        raise ModelConfigError(f"Unsupported model configuration provider: {provider!r}.")
    return provider


def get_ai_models_config() -> Dict[str, Any]:
    """Return the cached AI-models config object from the cloud config store."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    try:
        from cloud_lib import fetch_parameter_service_config

        _CONFIG_CACHE = fetch_parameter_service_config(
            "ai-models",
            region=os.environ.get("CLOUD_REGION"),
        )
    except Exception as exc:
        raise ModelConfigError("Could not load required ai-models configuration from the cloud config store.") from exc
    if not isinstance(_CONFIG_CACHE, dict):
        raise ModelConfigError("Required ai-models configuration from the cloud config store must be a JSON object.")
    return _CONFIG_CACHE


def get_model_default(
    category: str,
    provider: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Get a default model ID from ``model_defaults.<category>.<provider>``.

    Args:
        category: One of ``embedding``, ``llm``, ``vision``,
            ``graph_extraction``, ``summary``.
        provider: Provider-specific configuration namespace.
            Callers must resolve this explicitly (e.g. via
            :func:`resolve_config_provider_key`).
        fallback: Explicit value returned when the config lookup yields *None*.

    Returns:
        Model ID string, or *fallback* if not configured.
    """
    cfg = get_ai_models_config()
    entry = cfg.get("model_defaults", {}).get(category, {}).get(provider)
    if entry is None:
        return fallback
    # Support both string ("model-id") and object ({"model_id": ..., "region": ...})
    if isinstance(entry, dict):
        return entry.get("model_id") or fallback
    return entry or fallback


def get_model_spec(
    category: str,
    provider: str,
    fallback_model: Optional[str] = None,
    fallback_region: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Get model ID and region from ``model_defaults.<category>.<provider>``.

    Supports both string values and object format::

        # String form: region comes from fallback/environment
        "llm": {"provider": "provider-model-id"}

        # Object form: region is explicit per model
        "llm": {"provider": {"model_id": "provider-model-id", "region": "region-1"}}

    Args:
        category: One of ``embedding``, ``llm``, ``vision``,
            ``graph_extraction``, ``summary``.
        provider: Provider-specific configuration namespace.
            Callers must resolve this explicitly (e.g. via
            :func:`resolve_config_provider_key`).
        fallback_model: Model ID returned when not configured.
        fallback_region: Region returned when not configured (defaults to
            ``CLOUD_REGION`` env var or ``None``).

    Returns:
        Dict with keys ``model_id`` and ``region``.
    """
    cfg = get_ai_models_config()
    entry = cfg.get("model_defaults", {}).get(category, {}).get(provider)

    if fallback_region is None:
        fallback_region = os.environ.get("CLOUD_REGION")

    if entry is None:
        return {"model_id": fallback_model, "region": fallback_region}

    if isinstance(entry, dict):
        return {
            "model_id": entry.get("model_id") or fallback_model,
            "region": entry.get("region") or fallback_region,
        }

    # String format has no region specified.
    return {"model_id": entry or fallback_model, "region": fallback_region}


def get_vision_llm_models() -> Dict[str, Dict[str, Any]]:
    """Return the supported vision-LLM model configurations.

    Keyed by short name (``nova-pro``, ``claude-sonnet``, …).
    """
    cfg = get_ai_models_config()
    models = cfg.get("vision_llm_models", {})
    # Strip _description key if present
    return {k: v for k, v in models.items() if not k.startswith("_")}


def get_fallback_vision_llm_models() -> Dict[str, Dict[str, Any]]:
    """Return fallback vision-LLM model configurations from config.

    These models are used when primary config loading fails.
    Keyed by short name (``nova-pro``, ``claude-sonnet``, …).
    """
    cfg = get_ai_models_config()
    models = cfg.get("fallback_vision_llm_models", {})
    # Strip _description key if present
    return {k: v for k, v in models.items() if not k.startswith("_")}


def get_model_token_limits() -> Dict[str, int]:
    """Return model-family → token-limit mapping for summary generator."""
    cfg = get_ai_models_config()
    limits = cfg.get("model_token_limits", {})
    # Strip _description key if present
    return {k: v for k, v in limits.items() if not k.startswith("_")}


def get_inference_profile_required_models() -> set:
    """Return the set of model IDs that require inference-profile prefixing."""
    cfg = get_ai_models_config()
    models = cfg.get("inference_profile_required_models", {}).get("models", [])
    return set(models)


def resolve_inference_profile(model_id: str, region: str) -> str:
    """Deprecated compatibility wrapper around cloud_lib model resolution."""
    from cloud_lib.model_config import resolve_model_id

    try:
        return resolve_model_id(
            model_id,
            region,
            required_models=get_inference_profile_required_models(),
        )
    except ValueError as exc:
        raise ModelConfigError(str(exc)) from exc


def require_model_spec(
    category: str,
    provider: str,
) -> Dict[str, str]:
    """Strict version of :func:`get_model_spec` — raises if not configured.

    Use this in production code paths where a misconfigured model must
    cause an explicit failure instead of silently falling back to a
    hardcoded default.

    Returns:
        Dict with required keys ``model_id`` and ``region``.

    Raises:
        ModelConfigError: If the model_id or region is missing for the
            requested ``category`` / ``provider``.
    """
    spec = get_model_spec(
        category, provider,
        fallback_model=None,
        fallback_region=None,
    )
    missing = [k for k in ("model_id", "region") if not spec.get(k)]
    if missing:
        raise ModelConfigError(
            f"Required model config missing: model_defaults.{category}.{provider} "
            f"is missing {missing!r}. Configure it in the ai-models config store."
        )
    return spec


def get_embedding_supported_dimensions() -> Dict[str, list]:
    """Return model_id → list of supported vector dimensions.

    Loaded from ``model_defaults.embedding_supported_dimensions`` in the
    AI-models config.  Returns an empty dict when the section is absent.
    """
    cfg = get_ai_models_config()
    dims = cfg.get("model_defaults", {}).get("embedding_supported_dimensions", {})
    return {k: v for k, v in dims.items() if not k.startswith("_")}


def reload_config() -> Dict[str, Any]:
    """Clear the cache and reload the cloud config-store configuration."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return get_ai_models_config()


def seed_config(data: Dict[str, Any]) -> None:
    """Seed the in-memory config cache from an external source.

    Call this at pipeline startup to inject the ai-models config that was
    loaded from cloud config store or ConfigLoader.  This allows ``rag_lib`` adapters to
    resolve models without needing direct file access.

    Args:
        data: The ai-models config dict (the portion of the merged config
              that contains ``model_defaults``, ``vision_llm_models``, etc.).
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = data
    logger.info("model_config cache seeded externally (%d top-level keys)", len(data))


def refresh_config_from_cloud_config_store() -> Dict[str, Any]:
    """Re-fetch the ai-models config from cloud config store at runtime.

    This allows configuration changes (model IDs, regions, token limits)
    to take effect WITHOUT restarting the service.  The merged pipeline
    config on the caller side is NOT updated — only the ``rag_lib``
    model-config cache used by adapters.

    Returns:
        The refreshed ai-models config dict.

    Raises:
        RuntimeError: If the cloud config store fetch fails.
    """
    try:
        from cloud_lib import fetch_parameter_service_config

        data = fetch_parameter_service_config(
            "ai-models",
            region=os.environ.get("CLOUD_REGION"),
        )
    except Exception as exc:
        raise RuntimeError("Failed to refresh required ai-models configuration from the cloud config store.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Required ai-models configuration from the cloud config store must be a JSON object.")

    global _CONFIG_CACHE
    _CONFIG_CACHE = data
    logger.info(
        "model_config cache refreshed (%d top-level keys, source=config_store)",
        len(data),
    )
    return data
