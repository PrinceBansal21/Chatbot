from abc import ABC, abstractmethod

class PIIManager(ABC):
    """Interface for PII detection and redaction."""

    @abstractmethod
    def redact(self, text: str, **kwargs) -> str:
        """
        Redact PII from the given text.

        Args:
            text (str): The input text to redact.
            **kwargs: Additional arguments for the specific implementation.

        Returns:
            str: The text with PII redacted.
        """
        pass
