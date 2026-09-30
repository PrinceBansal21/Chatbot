class RAGException(Exception):
    """Base exception for all RAG library errors."""
    pass

class ResourceNotFoundException(RAGException):
    """Raised when a requested resource is not found."""
    pass

class PermissionDeniedException(RAGException):
    """Raised when the operation is not permitted."""
    pass

class ServiceUnavailableException(RAGException):
    """Raised when the underlying service is unavailable."""
    pass

class ConfigurationException(RAGException):
    """Raised when there is a configuration error."""
    pass

class TimeoutException(RAGException):
    """Raised when an operation times out."""
    pass
