from abc import ABC, abstractmethod
from typing import Optional, Any

class MarkdownManager(ABC):
    """
    High-Level Manager Interface for Markdown conversion operations.
    """

    @abstractmethod
    def convert(self, file_path: str, mime_type: Optional[str] = None, **kwargs) -> str:
        """
        Convert a document to Markdown.
        
        Args:
            file_path: Local path to the file to convert.
            mime_type: Optional MIME type of the file.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            The converted Markdown string.
        """
        pass
