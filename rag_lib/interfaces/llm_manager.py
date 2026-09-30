"""LLM Manager Interface for text generation tasks.

Provides abstraction for LLM-based operations like summarization,
question generation, and text completion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class LLMResponse:
    """Response from an LLM generation request.
    
    Attributes:
        text: The generated text content
        model_id: The model used for generation
        usage: Token usage statistics (input_tokens, output_tokens)
        metadata: Additional response metadata
    """
    text: str
    model_id: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMManager(ABC):
    """Interface for LLM-based text generation."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The user prompt/input text
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            
        Returns:
            LLMResponse with generated text and metadata
        """
        pass


__all__ = ["LLMManager", "LLMResponse"]
