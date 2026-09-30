"""Cloud-neutral RAG text generation adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_lib.interfaces.llm_manager import LLMManager, LLMResponse


class CloudLLMAdapter(LLMManager):
    """Adapt cloud_lib's normalized converse API to rag_lib's LLM contract."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        model_key: str = "llm",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        config_provider: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.model_key = model_key
        self.region = region
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self._manager_kwargs = kwargs
        self._adapter = None

    def _get_adapter(self):
        if self._adapter is None:
            from cloud_lib.factory import CloudFactory

            manager_kwargs = dict(self._manager_kwargs)
            manager_kwargs["model_key"] = self.model_key
            if self.model_id is not None:
                manager_kwargs["model_id"] = self.model_id
            self._adapter = CloudFactory.get_llm_manager(**manager_kwargs)
        return self._adapter

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        inference_config: Dict[str, Any] = {
            "maxTokens": max_tokens or self.default_max_tokens,
            "temperature": (
                temperature
                if temperature is not None
                else self.default_temperature
            ),
        }
        if stop_sequences:
            inference_config["stopSequences"] = stop_sequences

        response = self._get_adapter().converse(
            model_id=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system_prompt}] if system_prompt else None,
            inference_config=inference_config,
        )
        usage_info = response.get("usage", {})
        return LLMResponse(
            text=self._extract_text(response),
            model_id=self.model_id,
            usage={
                "input_tokens": usage_info.get("inputTokens", 0),
                "output_tokens": usage_info.get("outputTokens", 0),
            } if usage_info else {},
            metadata={
                "provider": "cloud",
                "model_key": self.model_key,
            },
        )

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        content = response.get("output", {}).get("message", {}).get("content", [])
        if not isinstance(content, list):
            return str(content) if content else ""
        return " ".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        ).strip()


__all__ = ["CloudLLMAdapter"]
