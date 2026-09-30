"""Shared low-level wrapper around the LLM **Converse** API.

Thin wrapper: all image handling, request construction, response parsing,
throttle back-off, and input-too-long retry logic live in cloud_lib's
LLM manager. This class only adds structured error logging before delegating
to ``CloudFactory.get_llm_manager()``.
"""
from __future__ import annotations

from typing import Any, Optional

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


class LLMConverseClient:
    """Thin, reusable wrapper around ``CloudFactory.get_llm_manager().converse_vision()``.

    Features
    --------
    * **Lazy adapter init** — the cloud_lib LLM adapter is created on first use.
        * **Provider model resolution** — delegates model selection to cloud_lib.
    * **Structured error logging** — original vs resolved model_id, region, error
      code, and payload characteristics for production debugging.

    Image handling (format normalisation, WebP conversion, compression/resize),
    Converse request/response construction, throttle retry, and input resizing
    are implemented by the selected cloud_lib LLM manager.
    """

    # Defaults -----------------------------------------------------------------
    DEFAULT_MAX_TOKENS = 256
    DEFAULT_TEMPERATURE = 0.2

    def __init__(
        self,
        region: Optional[str] = None,
        provider: Optional[str] = None,
        model_key: str = "llm",
    ) -> None:
        self.provider = provider
        self.region = region
        self.model_key = model_key
        self._adapter: Any = None  # lazy cloud_lib LLMManager

    # -- lazy adapter ------------------------------------------------------------

    @property
    def adapter(self) -> Any:
        """Return (and lazily create) the cloud_lib LLM adapter."""
        if self._adapter is None:
            from cloud_lib.factory import CloudFactory

            self._adapter = CloudFactory.get_llm_manager(model_key=self.model_key)
        return self._adapter

    # -- public API ------------------------------------------------------------

    def converse(
        self,
        *,
        model_id: Optional[str] = None,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_format: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Optional[str]:
        """Call the LLM Converse API (via cloud_lib) and return the extracted text.

        Parameters
        ----------
        model_id:
            Optional provider model or deployment identifier.
        prompt:
            The user-message text.
        image_bytes:
            Raw image bytes to attach as a vision payload.  When *None* the
            request is text-only.
        image_format:
            Format hint (``"png"``, ``"jpeg"``, …); normalised by cloud_lib.
        system_prompt:
            Optional system message prepended to the conversation.
        max_tokens / temperature:
            Inference configuration overrides.
        """
        resolved_model_id = model_id
        try:
            return self.adapter.converse_vision(
                model_id=model_id,
                prompt=prompt,
                image_bytes=image_bytes,
                image_format=image_format,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            self._log_llm_error(
                exc,
                original_model_id=model_id,
                resolved_model_id=resolved_model_id,
                has_image=image_bytes is not None,
                prompt_len=len(prompt),
            )
            raise

    # -- error logging -----------------------------------------------------------

    def _log_llm_error(
        self,
        exc: Exception,
        *,
        original_model_id: Optional[str],
        resolved_model_id: Optional[str],
        has_image: bool,
        prompt_len: int,
    ) -> None:
        """Emit a structured error log entry for a failed LLM invocation.

        Includes everything needed to diagnose the failure in production:
        original vs resolved model_id, region, error code/message, and the
        type of payload.
        """
        error_code = getattr(exc, "error_code", "") or ""

        prefix_applied = (
            resolved_model_id is not None
            and
            resolved_model_id != original_model_id
            and resolved_model_id.startswith(("us.", "eu.", "ap.", "global."))
        )

        LOGGER.log_warning(
            ProcessLogger.STAGE_CONVERT,
            (
                "LLM Converse call FAILED | "
                f"region={self.region} | "
                f"original_model_id={original_model_id!r} | "
                f"resolved_model_id={resolved_model_id!r} | "
                f"prefix_applied={prefix_applied} | "
                f"error_type={type(exc).__name__} | "
                f"error_code={error_code!r} | "
                f"has_image={has_image} | "
                f"prompt_chars={prompt_len} | "
                f"message={str(exc)[:500]}"
            ),
        )

