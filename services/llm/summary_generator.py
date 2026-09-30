"""
Summary Generator Service

Generates page/document summaries using LLM for improved retrieval.
The summary is attached to each chunk's metadata, providing document-level
context that helps with semantic matching during retrieval.

Benefits:
- Better semantic matching for queries
- Document-level context in each chunk
- Improved reranking capabilities
- Reduced hallucination through grounded context
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import ProcessLogger, get_logger
from .llm_converse_client import LLMConverseClient

LOGGER = get_logger(__name__)

# Approximate max input tokens per model family, loaded from configuration.
def _load_model_token_limits() -> Dict[str, int]:
    """Load token limits from central configuration."""
    try:
        from rag_lib.model_config import get_model_token_limits
        limits = get_model_token_limits()
        if limits:
            return limits
    except Exception:
        pass
    return {}


_MODEL_TOKEN_LIMITS: Dict[str, int] = _load_model_token_limits()

# Rough chars-per-token ratio (conservative: 1 token ≈ 3.5 chars)
_CHARS_PER_TOKEN = 3.5


class SummaryGenerator:
    """
    Service for generating document summaries using LLM.
    
    The generated summary is embedded in each chunk's metadata, enabling:
    - Multi-level retrieval (query summaries first, then chunks)
    - Better semantic matching for broad queries
    - Document context preservation across chunks
    
    Args:
        config: Summary generation configuration dict
        region: Cloud region for LLM client
        
    Configuration options:
        enabled: Whether summary generation is enabled (default: True)
        model_id: LLM model ID for summarization
        max_input_chars: Maximum chars to send to LLM (default: 50000)
        max_summary_tokens: Maximum tokens for summary (default: 300)
        style: Summary style - 'concise', 'detailed', 'bullet' (default: 'concise')
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that creates accurate, informative document summaries.
Your summaries should capture:
- The main topic and purpose of the document
- Key facts, concepts, and important details
- The target audience or use case if apparent

Be concise but comprehensive. Do not include phrases like "This document" or "The article discusses"."""

    DEFAULT_TEMPLATES = {
        "concise": """Summarize the following document in 2-4 sentences, capturing the main topic and key points:

{text}

Summary:""",
        
        "detailed": """Provide a comprehensive summary of the following document, preserving important details:

{text}

Detailed Summary:""",
        
        "bullet": """Summarize the following document as a bullet-point list of key facts (5-10 points):

{text}

Key Points:""",
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        region: Optional[str] = None,
    ):
        """Initialize summary generation using the logical ``summary`` model role."""
        self.config = config
        self.enabled = config.get("enabled", True)
        self.model_id = config.get("model_id")
        self.model_label = self.model_id or "summary"

        self.max_input_chars = config.get("max_input_chars", 50000)
        self.max_summary_tokens = config.get("max_summary_tokens", 300)
        self.style = config.get("style", "concise")
        self.temperature = config.get("temperature", 0.2)
        
        # Custom prompts from config
        self.system_prompt = config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        self.templates = {**self.DEFAULT_TEMPLATES, **config.get("templates", {})}
        
        self._converse = LLMConverseClient(model_key="summary")
        
        # Compute model-aware safe character limit
        self._safe_max_chars = self._compute_safe_max_chars()
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT, 
            f"SummaryGenerator initialized: enabled={self.enabled}, model={self.model_label}, "
            f"style={self.style}, safe_max_chars={self._safe_max_chars:,}"
        )

    def _compute_safe_max_chars(self) -> int:
        """Return the safe maximum input characters for the configured model.

        Uses ``_MODEL_TOKEN_LIMITS`` to find a token ceiling, converts to
        characters, then takes the **minimum** of that and the user-configured
        ``max_input_chars`` so we never exceed the model's context window.
        """
        model_lower = (self.model_id or "").lower()
        token_limit: Optional[int] = None
        for fragment, limit in _MODEL_TOKEN_LIMITS.items():
            if fragment in model_lower:
                token_limit = limit
                break

        if token_limit is None:
            # Unknown model — fall back to a conservative 100K tokens
            token_limit = 100_000
            LOGGER.log_stage(
                ProcessLogger.STAGE_INIT,
                f"Model '{self.model_label}' not in token-limit map; "
                f"using conservative fallback of {token_limit:,} tokens"
            )

        # Leave room for system prompt + template overhead (~2000 tokens)
        overhead_tokens = 2_000
        usable_tokens = max(token_limit - overhead_tokens, 10_000)
        char_limit = int(usable_tokens * _CHARS_PER_TOKEN)

        # Never exceed user-configured max_input_chars
        safe = min(char_limit, self.max_input_chars)
        return safe
    
    def generate_summary(
        self,
        markdown: str,
        *,
        style: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a summary for the document.
        
        Args:
            markdown: Full markdown content of the document
            style: Override the default style ('concise', 'detailed', 'bullet')
            file_name: Optional file name for logging
            
        Returns:
            Summary string or None if generation fails
        """
        if not self.enabled:
            LOGGER.log_stage(ProcessLogger.STAGE_CONVERT, "Summary generation disabled")
            return None
        
        if not markdown or len(markdown.strip()) < 100:
            LOGGER.log_stage(ProcessLogger.STAGE_CONVERT, "Document too short for summarization")
            return None
        
        style = style or self.style
        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT, 
            f"Generating {style} summary for document ({len(markdown):,} chars)"
        )
        
        try:
            # Truncate to model-safe character limit
            text = markdown[:self._safe_max_chars]
            if len(markdown) > self._safe_max_chars:
                LOGGER.log_stage(
                    ProcessLogger.STAGE_CONVERT,
                    f"Truncated document from {len(markdown):,} to {self._safe_max_chars:,} chars "
                    f"(model limit for {self.model_label})"
                )
            
            # Get template for style
            template = self.templates.get(style, self.templates["concise"])
            prompt = template.format(text=text)
            
            # Call the LLM (with automatic retry on input-too-long)
            summary = self._invoke_llm(prompt)
            
            if summary:
                LOGGER.log_stage(
                    ProcessLogger.STAGE_CONVERT,
                    f"Generated summary: {len(summary)} chars"
                )
            
            return summary
            
        except Exception as e:
            LOGGER.log_error(f"Summary generation failed: {e}", exception=e)
            return None
    
    def _invoke_llm(self, prompt: str) -> Optional[str]:
        """Invoke the configured LLM for summarization using the shared Converse client.

        Delegates the LLM request to ``LLMConverseClient``.
        """
        return self._converse.converse(
            model_id=self.model_id,
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_summary_tokens,
            temperature=self.temperature,
        )
    
    def add_summary_to_metadata(
        self,
        metadata: Dict[str, str],
        summary: Optional[str],
    ) -> Dict[str, str]:
        """
        Add the summary to metadata dict.
        
        Args:
            metadata: Base metadata dictionary
            summary: Document summary (or None)
            
        Returns:
            Updated metadata dictionary
        """
        if summary:
            metadata["document_summary"] = summary
            metadata["has_summary"] = "true"
        else:
            metadata["has_summary"] = "false"
        
        return metadata

    # ── Page-level summaries ──────────────────────────────────────

    PAGE_SUMMARY_TEMPLATE = (
        "Summarize the following page content in 1-2 sentences, "
        "capturing the key information on this page:\n\n{text}\n\nPage Summary:"
    )

    def generate_page_summaries(
        self,
        markdown: str,
        max_workers: int = 5,
    ) -> Dict[str, str]:
        """
        Generate a short summary for each page in the document.

        Pages are detected via ``<!-- page_number: N -->`` markers.
        If there are no markers (e.g. webpages), the whole text is treated
        as a single page and an empty dict is returned (the document-level
        summary already covers it).
        
        Uses ThreadPoolExecutor for parallel LLM calls (I/O-bound).

        Args:
            markdown: Full markdown content with page markers
            max_workers: Number of parallel summary workers (default: 5)

        Returns:
            Mapping of page number (str) → page summary text.
        """
        if not self.enabled:
            return {}

        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        page_pattern = re.compile(
            r"<!--\s*page_number:\s*(\d+)\s*-->", re.IGNORECASE
        )
        parts = page_pattern.split(markdown)

        # parts layout: [preamble, page_num_1, content_1, page_num_2, content_2, ...]
        if len(parts) < 3:
            # No page markers found — nothing to summarise per-page
            return {}

        # Collect pages to summarize
        pages_to_summarize = []
        for i in range(1, len(parts), 2):
            page_num = parts[i]
            page_content = parts[i + 1].strip()
            if not page_content or len(page_content) < 80:
                continue
            pages_to_summarize.append((str(page_num), page_content))
        
        if not pages_to_summarize:
            return {}
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_CONVERT,
            f"Generating page summaries for {len(pages_to_summarize)} pages with {max_workers} parallel workers"
        )

        page_summaries: Dict[str, str] = {}
        
        def _summarize_page(page_num: str, content: str) -> tuple:
            """Summarize a single page, returning (page_num, summary_or_None)."""
            try:
                text = content[: self._safe_max_chars]
                prompt = self.PAGE_SUMMARY_TEMPLATE.format(text=text)
                summary = self._invoke_llm(prompt)
                return page_num, summary
            except Exception:
                return page_num, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_summarize_page, pn, content): pn
                for pn, content in pages_to_summarize
            }
            for future in as_completed(futures):
                page_num, summary = future.result()
                if summary:
                    page_summaries[page_num] = summary

        return page_summaries


__all__ = ["SummaryGenerator"]
