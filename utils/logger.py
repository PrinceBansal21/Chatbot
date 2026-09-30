"""Structured logging utilities with file_id context tracking.

Provides:
- ContextVar-based file_id tracking (works in asyncio / main thread)
- Thread-safe fallback via threading.local for worker threads that lose ContextVar
- ProcessLogger wrapper for structured stage-based logging
- JSON and human-readable formatters with automatic file_id injection
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Context variable to store the current file_id.
# ContextVars work perfectly in asyncio and in the *same* thread, but they
# are NOT automatically inherited by new threads created via
# ``concurrent.futures.ThreadPoolExecutor``.  To handle that we keep a
# secondary module-level ``_global_file_id`` that acts as a fallback for
# threads spawned by third-party libraries (e.g., rag_lib's parallel_hybrid_pdf
# adapter) where we cannot wrap the submitted callables.
#
# For our own thread pools (worker.py) we call ``set_file_id()`` inside
# each thread's processing function, which sets both stores.
# ---------------------------------------------------------------------------
_current_file_id: ContextVar[Optional[str]] = ContextVar("current_file_id", default=None)

# Module-level fallback for threads that don't inherit ContextVar.
# This is safe because the ingestion pipeline processes one file_id at a time
# in the main thread; child threads read this value only.
_global_file_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Module-level event buffer for ingestion log accumulation.
# Events are accumulated during processing and flushed to object storage at the end.
# Thread-safe via a lock since child threads may also log events.
# ---------------------------------------------------------------------------
_event_buffer: List[Dict[str, Any]] = []
_event_buffer_lock = threading.Lock()


def set_file_id(file_id: Optional[str]) -> None:
    """Set the current file_id for logging context (ContextVar + global fallback)."""
    global _global_file_id
    _current_file_id.set(file_id)
    _global_file_id = file_id


def get_file_id() -> Optional[str]:
    """Get the current file_id from logging context.

    Checks ContextVar first (preferred), falls back to module-level global
    so that third-party library threads still carry the file_id.
    """
    fid = _current_file_id.get()
    if fid is not None:
        return fid
    return _global_file_id


def clear_file_id() -> None:
    """Clear the current file_id from logging context."""
    global _global_file_id
    _current_file_id.set(None)
    _global_file_id = None


def get_ingestion_events() -> List[Dict[str, Any]]:
    """Return a copy of accumulated ingestion log events."""
    with _event_buffer_lock:
        return list(_event_buffer)


def clear_ingestion_events() -> None:
    """Clear the ingestion event buffer."""
    with _event_buffer_lock:
        _event_buffer.clear()


def _append_event(event: Dict[str, Any]) -> None:
    """Append an event to the ingestion event buffer (thread-safe)."""
    with _event_buffer_lock:
        _event_buffer.append(event)


class FileIdContextFilter(logging.Filter):
    """Logging filter that adds file_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.file_id = get_file_id() or "N/A"  # type: ignore[attr-defined]
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with file_id context."""

    def format(self, record: logging.LogRecord) -> str:
        file_id = getattr(record, "file_id", "N/A")
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "file_id": file_id,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            log_entry["data"] = extra_data
            
        return json.dumps(log_entry)


class ProcessLogger:
    """
    Logger wrapper that provides structured logging with file_id tracking.
    
    This logger is designed to trace the entire document processing flow
    from input to output with consistent file_id context.
    """

    # Processing stages for consistent logging
    STAGE_INIT = "INITIALIZATION"
    STAGE_INPUT = "INPUT_VALIDATION"
    STAGE_FETCH = "DOCUMENT_FETCH"
    STAGE_CONVERT = "MARKDOWN_CONVERSION"
    STAGE_VISION = "IMAGE_DESCRIPTION"
    STAGE_PII = "PII_REDACTION"
    STAGE_STORAGE = "MARKDOWN_STORAGE"
    STAGE_CHUNK = "CHUNKING"
    STAGE_EMBED = "EMBEDDING"
    STAGE_INDEX = "VECTOR_INDEXING"
    STAGE_DB_UPDATE = "DATABASE_UPDATE"
    STAGE_CLEANUP = "CLEANUP"
    STAGE_COMPLETE = "PROCESSING_COMPLETE"
    STAGE_ERROR = "ERROR"
    STAGE_DELETE = "DELETE_OPERATION"

    def __init__(self, name: str, use_json: bool = True):
        """
        Initialize the ProcessLogger.
        
        Args:
            name: Logger name (typically __name__ of the module)
            use_json: If True, use JSON structured logging; otherwise use human-readable format
        """
        self.logger = logging.getLogger(name)
        self._setup_handler(use_json)
        
    def _setup_handler(self, use_json: bool) -> None:
        """Set up logging handler with appropriate formatter."""
        # Add file_id context filter
        if not any(isinstance(f, FileIdContextFilter) for f in self.logger.filters):
            self.logger.addFilter(FileIdContextFilter())

    def _log(
        self,
        level: int,
        stage: str,
        message: str,
        file_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Internal logging method with structured context.
        
        Args:
            level: Logging level (e.g., logging.INFO)
            stage: Processing stage identifier
            message: Log message
            file_id: Optional file_id override (uses context var if not provided)
            extra_data: Additional structured data to include
        """
        current_file_id = file_id or get_file_id() or "N/A"
        
        # Format message with stage and file_id prefix for easy grep/filtering
        formatted_msg = f"[{stage}] [file_id={current_file_id}] {message}"
        
        extra = {"extra_data": extra_data} if extra_data else {}
        self.logger.log(level, formatted_msg, extra=extra)

        # Accumulate event for ingestion log (only INFO+ to avoid noise)
        if level >= logging.INFO:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": logging.getLevelName(level),
                "stage": stage,
                "message": message,
                "file_id": current_file_id,
            }
            if extra_data:
                event["data"] = extra_data
            _append_event(event)

    def start_processing(self, file_id: str, payload: Dict[str, Any]) -> None:
        """Log the start of document processing."""
        set_file_id(file_id)
        self._log(
            logging.INFO,
            self.STAGE_INPUT,
            f"Starting document processing",
            extra_data={
                "operation": payload.get("operation", "index"),
                "source_uri": payload.get("source_uri") or payload.get("storage_uri") or payload.get("storage_location") or payload.get("web_url"),
                "file_type": payload.get("file_type"),
                "target_index": payload.get("index_name"),
            }
        )

    def end_processing(self, file_id: str, result: Dict[str, Any]) -> None:
        """Log the completion of document processing with a structured summary."""
        # Extract key metrics for a concise, structured log line
        chunks_created = result.get("chunks_created", 0)
        text_chunks = result.get("text_chunks_created", 0)
        image_chunks = result.get("image_chunks_created", 0)
        infographic_chunks = result.get("infographic_chunks_created", 0)
        chunks_deleted = result.get("chunks_deleted", 0)
        image_count = result.get("image_count", 0)
        index_names = result.get("index_names") or [result.get("index_name", "N/A")]
        config = result.get("configuration", {})

        summary_parts = [
            f"chunks_created={chunks_created}",
            f"text={text_chunks}",
            f"image={image_chunks}",
            f"infographic={infographic_chunks}",
            f"deleted={chunks_deleted}",
            f"images_extracted={image_count}",
            f"indices={index_names}",
        ]
        if result.get("markdown_location"):
            summary_parts.append(f"markdown={result['markdown_location']}")
        if config:
            summary_parts.append(f"md_provider={config.get('markdown_provider', 'N/A')}")
            summary_parts.append(f"chunking={config.get('chunking_strategy', 'N/A')}")

        self._log(
            logging.INFO,
            self.STAGE_COMPLETE,
            f"Ingestion complete: {', '.join(summary_parts)}",
            extra_data=result,
        )

        # Per-index breakdown (if multi-index)
        indexed_per_index = result.get("indexed_per_index")
        if indexed_per_index and len(indexed_per_index) > 1:
            for idx_name, count in indexed_per_index.items():
                self._log(
                    logging.INFO,
                    self.STAGE_COMPLETE,
                    f"Index '{idx_name}': {count} chunks indexed",
                )

        clear_file_id()

    def log_stage(
        self,
        stage: str,
        message: str,
        level: int = logging.INFO,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a processing stage event."""
        self._log(level, stage, message, extra_data=extra_data)

    def log_fetch(self, source_uri: str, local_path: Optional[str] = None) -> None:
        """Log document fetch operation."""
        self._log(
            logging.INFO,
            self.STAGE_FETCH,
            f"Fetching document from: {source_uri}",
            extra_data={"source_uri": source_uri, "local_path": local_path}
        )

    def log_fetch_complete(self, document_path: str) -> None:
        """Log document fetch completion."""
        self._log(
            logging.INFO,
            self.STAGE_FETCH,
            f"Document fetched successfully: {document_path}"
        )

    def log_conversion(self, provider: Optional[str], document_path: str) -> None:
        """Log markdown conversion start."""
        self._log(
            logging.INFO,
            self.STAGE_CONVERT,
            f"Converting document to markdown using provider: {provider or 'default'}",
            extra_data={"provider": provider, "document_path": document_path}
        )

    def log_conversion_complete(self, content_length: int) -> None:
        """Log markdown conversion completion."""
        self._log(
            logging.INFO,
            self.STAGE_CONVERT,
            f"Markdown conversion completed, content length: {content_length} chars"
        )

    def log_vision(self, provider: Optional[str], image_count: int) -> None:
        """Log image description processing."""
        self._log(
            logging.INFO,
            self.STAGE_VISION,
            f"Processing {image_count} images for description using provider: {provider or 'default'}"
        )

    def log_vision_complete(self, images_processed: int) -> None:
        """Log image description completion."""
        self._log(
            logging.INFO,
            self.STAGE_VISION,
            f"Image description completed for {images_processed} images"
        )

    def log_pii_redaction(self, enabled: bool) -> None:
        """Log PII redaction status."""
        if enabled:
            self._log(logging.INFO, self.STAGE_PII, "Performing PII redaction on markdown content")
        else:
            self._log(logging.DEBUG, self.STAGE_PII, "PII redaction is disabled")

    def log_pii_complete(self) -> None:
        """Log PII redaction completion."""
        self._log(logging.INFO, self.STAGE_PII, "PII redaction completed")

    def log_storage(self, bucket: str, key: str) -> None:
        """Log markdown storage operation."""
        self._log(
            logging.INFO,
            self.STAGE_STORAGE,
            f"Storing markdown to storage",
            extra_data={"bucket": bucket, "key": key}
        )

    def log_chunking(self, strategy: Optional[str]) -> None:
        """Log chunking operation start."""
        self._log(
            logging.INFO,
            self.STAGE_CHUNK,
            f"Chunking markdown content using strategy: {strategy or 'default'}"
        )

    def log_chunking_complete(self, chunk_count: int) -> None:
        """Log chunking completion."""
        self._log(
            logging.INFO,
            self.STAGE_CHUNK,
            f"Chunking completed, created {chunk_count} chunks"
        )

    def log_embedding(self, provider: Optional[str], chunk_count: int) -> None:
        """Log embedding generation start."""
        self._log(
            logging.INFO,
            self.STAGE_EMBED,
            f"Generating embeddings for {chunk_count} chunks using provider: {provider or 'default'}"
        )

    def log_embedding_complete(self, embedding_count: int) -> None:
        """Log embedding completion."""
        self._log(
            logging.INFO,
            self.STAGE_EMBED,
            f"Embedding generation completed for {embedding_count} chunks"
        )

    def log_indexing(self, index_name: Optional[str], chunk_count: int) -> None:
        """Log vector indexing operation."""
        self._log(
            logging.INFO,
            self.STAGE_INDEX,
            f"Indexing {chunk_count} chunks to vector database",
            extra_data={"index_name": index_name or "default", "chunk_count": chunk_count}
        )

    def log_indexing_complete(self, index_name: Optional[str]) -> None:
        """Log indexing completion."""
        self._log(
            logging.INFO,
            self.STAGE_INDEX,
            f"Vector indexing completed to index: {index_name or 'default'}"
        )

    def log_delete(self, index_name: Optional[str], chunks_deleted: int) -> None:
        """Log delete operation."""
        if chunks_deleted > 0:
            self._log(
                logging.INFO,
                self.STAGE_DELETE,
                f"Deleted {chunks_deleted} existing chunks from index: {index_name or 'default'}"
            )
        else:
            self._log(
                logging.DEBUG,
                self.STAGE_DELETE,
                f"No existing chunks found in index: {index_name or 'default'}"
            )

    def log_db_update(self, status: str) -> None:
        """Log database status update."""
        self._log(
            logging.INFO,
            self.STAGE_DB_UPDATE,
            f"Updated database status to: {status}"
        )

    def log_chunk_validation_summary(
        self, 
        total_chunks: int, 
        valid_count: int, 
        invalid_count: int,
        warning_count: int
    ) -> None:
        """Log chunk validation summary."""
        self._log(
            logging.INFO,
            self.STAGE_INDEX,
            f"Chunk validation: {valid_count}/{total_chunks} valid, {invalid_count} invalid, {warning_count} warnings",
            extra_data={
                "total_chunks": total_chunks,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "warning_count": warning_count
            }
        )

    def log_chunk_validation_warning(self, chunk_id: str, content_length: int, reason: Optional[str] = None) -> None:
        """Log warning for chunk approaching length limits."""
        self._log(
            logging.WARNING,
            self.STAGE_INDEX,
            f"Chunk length warning for {chunk_id}: {content_length} chars",
            extra_data={
                "chunk_id": chunk_id,
                "content_length": content_length,
                "reason": reason
            }
        )

    def log_chunk_validation_error(self, chunk_id: str, content_length: int, reason: Optional[str] = None) -> None:
        """Log error for chunk failing validation."""
        self._log(
            logging.ERROR,
            self.STAGE_INDEX,
            f"Chunk validation failed for {chunk_id}: {reason}",
            extra_data={
                "chunk_id": chunk_id,
                "content_length": content_length,
                "reason": reason
            }
        )

    def log_cleanup(self) -> None:
        """Log cleanup operation."""
        self._log(logging.DEBUG, self.STAGE_CLEANUP, "Cleaning up temporary files")

    def log_error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log an error with optional exception details."""
        data = extra_data or {}
        if exception:
            data["error_type"] = type(exception).__name__
            data["error_message"] = str(exception)
        self._log(logging.ERROR, self.STAGE_ERROR, message, extra_data=data)

    def log_warning(self, stage: str, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, stage, message, extra_data=extra_data)

    def log_debug(self, stage: str, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, stage, message, extra_data=extra_data)


def configure_root_logger(level: int = logging.INFO, use_json: bool = False) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level
        use_json: If True, use JSON structured formatting
    """
    root_logger = logging.getLogger()
    
    # Remove existing handlers from root logger to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if use_json:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(file_id)s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
    
    handler.addFilter(FileIdContextFilter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Configure third-party library loggers (e.g., docling)
    configure_third_party_loggers(level)


def configure_third_party_loggers(level: int = logging.INFO) -> None:
    """
    Configure logging for third-party libraries.
    
    - Adds ``FileIdContextFilter`` to rag_lib and docling loggers so they
      carry the active ``file_id`` in every log record.
    - Suppresses verbose / noisy libraries to WARNING to reduce log clutter.
    - Clears existing handlers to prevent duplicate log messages in notebooks.
    
    Args:
        level: Logging level for application-relevant third-party loggers
    """
    # rag_lib loggers — these are our own library, keep at application level
    # and ensure file_id propagates
    rag_lib_loggers = [
        "rag_lib",
        "rag_lib.adapters",
        "rag_lib.adapters.markdown",
        "rag_lib.adapters.markdown.parallel_hybrid_pdf",
        "rag_lib.adapters.markdown.parallel_hybrid_pdf.markdown_adapter",
        "rag_lib.adapters.markdown.docling",
        "rag_lib.adapters.markdown.docling.markdown_adapter",
        "rag_lib.adapters.markdown.vision_llm",
        "rag_lib.adapters.markdown.vision_llm.markdown_adapter",
        "rag_lib.adapters.chunking",
        "rag_lib.adapters.embedding",
    ]
    
    for logger_name in rag_lib_loggers:
        logger = logging.getLogger(logger_name)
        # Clear existing handlers to prevent duplicates (especially in notebooks)
        logger.handlers.clear()
        logger.setLevel(level)
        # Ensure propagation to root logger (where our single handler lives)
        logger.propagate = True
        if not any(isinstance(f, FileIdContextFilter) for f in logger.filters):
            logger.addFilter(FileIdContextFilter())
    
    # Docling library loggers — keep at WARNING to reduce noise
    # (detailed docling logs are rarely useful for debugging ingestion)
    docling_loggers = [
        "docling",
        "docling.document_converter",
        "docling.datamodel",
        "docling.datamodel.document",
        "docling.backend",
        "docling.pipeline",
        "docling.pipeline.base_pipeline",
        "docling.models",
        "docling.models.auto_ocr_model",
        "docling.models.rapid_ocr_model",
        "docling.models.factories",
        "docling.models.factories.base_factory",
        "docling.utils",
        "docling.utils.accelerator_utils",
        "docling_core",
        "docling_ibm_models",
    ]
    
    for logger_name in docling_loggers:
        logger = logging.getLogger(logger_name)
        # Clear existing handlers to prevent duplicates
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        logger.propagate = True
        if not any(isinstance(f, FileIdContextFilter) for f in logger.filters):
            logger.addFilter(FileIdContextFilter())
    
    from cloud_lib import configure_provider_loggers

    configure_provider_loggers(level=logging.WARNING)

    # Suppress verbose document-processing libraries to WARNING or higher
    verbose_loggers = [
        "RapidOCR",           # RapidOCR model loading noise
        "rapidocr",
        "onnxruntime",        # ONNX runtime device discovery
        "PIL",
        "matplotlib",
    ]
    for logger_name in verbose_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        logger.propagate = True


def get_logger(name: str) -> ProcessLogger:
    """Get a ProcessLogger instance for the given module name."""
    return ProcessLogger(name)


def reset_logging() -> None:
    """
    Reset all logging configuration to prevent duplicate log messages.
    
    Call this at the start of a Jupyter notebook cell before initializing
    the Pipeline to ensure clean logging state. This is especially useful
    when re-running cells in notebooks.
    
    Usage in notebooks:
        from utils.logger import reset_logging
        reset_logging()
        
        from app import Pipeline
        pipeline = Pipeline()
    """
    # Clear all handlers from all existing loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
    
    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Reconfigure with default settings
    configure_root_logger(level=logging.INFO, use_json=False)
