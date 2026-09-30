"""VectorDB indexing service."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from cloud_lib.factory import CloudFactory
from cloud_lib.exceptions import CloudlibException
from rag_lib.interfaces.chunking_manager import Chunk

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)


@dataclass
class ChunkValidationResult:
    """Result of chunk length validation."""
    is_valid: bool
    chunk_id: str
    content_length: int
    reason: Optional[str] = None


class VectorDBIndexer:
    """Handles vector upserts into VectorDB."""

    # Default chunk length constraints (in characters)
    DEFAULT_MIN_CHUNK_LENGTH = 10
    DEFAULT_MAX_CHUNK_LENGTH = 32000  # Conservative vector database text limit
    DEFAULT_WARN_CHUNK_LENGTH = 16000  # Warn when chunks exceed this length
    
    # Default core fields that are always stored with chunks
    DEFAULT_CORE_FIELDS = ["chunk_id", "file_id", "content", "order", "vector"]

    def __init__(self, config: Dict, dependency_index: Optional[str] = None, chunk_fields_config: Optional[Dict] = None):
        self.index = config.get("index") or dependency_index
        if not self.index:
            raise ValueError("VectorDB index not configured; update config or deployment dependencies")
        self.region = config.get("region") or os.environ.get("CLOUD_REGION")

        # Connection timeout configuration
        # Bulk indexing configuration
        self.bulk_chunk_size = config.get("bulk_chunk_size", 500)
        self.bulk_request_timeout = config.get("bulk_request_timeout", 120)

        self.vectordb = CloudFactory.get_vectordb_manager()

        if "allow_custom_ids" in config:
            self.allow_custom_ids = bool(config["allow_custom_ids"])
        else:
            self.allow_custom_ids = not self.vectordb.is_serverless

        # Chunk length validation configuration
        validation_config = config.get("chunk_validation", {})
        self.validate_chunk_length = validation_config.get("enabled", True)
        self.min_chunk_length = validation_config.get("min_length", self.DEFAULT_MIN_CHUNK_LENGTH)
        self.max_chunk_length = validation_config.get("max_length", self.DEFAULT_MAX_CHUNK_LENGTH)
        self.warn_chunk_length = validation_config.get("warn_length", self.DEFAULT_WARN_CHUNK_LENGTH)
        self.skip_invalid_chunks = validation_config.get("skip_invalid", False)
        self.truncate_long_chunks = validation_config.get("truncate_long", False)
        
        # Chunk fields configuration - defines what fields are stored with each chunk type
        self.chunk_fields_config = chunk_fields_config or {}
        self._init_chunk_fields()
    
    def _init_chunk_fields(self) -> None:
        """Initialize chunk field mappings from configuration."""
        # Extract field names from configuration sections
        self.core_fields = self._extract_field_names(
            self.chunk_fields_config.get("core_fields", {}),
            default=self.DEFAULT_CORE_FIELDS
        )
        self.metadata_fields = self._extract_field_names(
            self.chunk_fields_config.get("metadata_fields", {})
        )
        self.text_chunk_fields = self._extract_field_names(
            self.chunk_fields_config.get("text_chunk_fields", {})
        )
        self.image_chunk_fields = self._extract_field_names(
            self.chunk_fields_config.get("image_chunk_fields", {})
        )
        self.summary_chunk_fields = self._extract_field_names(
            self.chunk_fields_config.get("summary_chunk_fields", {})
        )
        self.contextual_chunk_fields = self._extract_field_names(
            self.chunk_fields_config.get("contextual_chunk_fields", {})
        )
        
        LOGGER.log_debug(
            ProcessLogger.STAGE_INIT,
            f"Chunk fields initialized - core: {len(self.core_fields)}, metadata: {len(self.metadata_fields)}, "
            f"text: {len(self.text_chunk_fields)}, contextual: {len(self.contextual_chunk_fields)}, image: {len(self.image_chunk_fields)}"
        )
    
    @staticmethod
    def _extract_field_names(config_section: Dict, default: List[str] = None) -> List[str]:
        """Extract field names from a config section, excluding description keys."""
        if not config_section:
            return default or []
        return [k for k in config_section.keys() if not k.startswith("_")]
    
    def get_configured_fields_for_chunk_type(self, chunk_type: str) -> List[str]:
        """
        Get the list of configured fields for a specific chunk type.
        
        Args:
            chunk_type: Type of chunk ('text', 'image', 'summary', 'page', 'whole_document')
            
        Returns:
            List of field names that should be stored for this chunk type
        """
        # Start with core fields (always included)
        fields = set(self.core_fields)
        
        # Add metadata fields
        fields.update(self.metadata_fields)
        
        # Add type-specific fields
        if chunk_type == "image":
            fields.update(self.image_chunk_fields)
        elif chunk_type == "summary":
            fields.update(self.summary_chunk_fields)
        else:
            # Default text chunk
            fields.update(self.text_chunk_fields)
        
        # Always include contextual fields — they are additive and only
        # populated when the contextual chunking strategy is used.
        fields.update(self.contextual_chunk_fields)
        
        return list(fields)
    
    def build_document_body(
        self, 
        chunk: Chunk, 
        vector: List[float],
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build the document body for indexing based on configured fields.
        
        Args:
            chunk: The chunk to build document body for
            vector: The embedding vector
            content: Optional content override (e.g., truncated content)
            
        Returns:
            Dictionary containing the document body with configured fields
        """
        # Determine chunk type from metadata
        chunk_type = chunk.metadata.get("chunk_type", "text")
        
        # Get configured fields for this chunk type
        configured_fields = self.get_configured_fields_for_chunk_type(chunk_type)
        
        # Build base document with core fields
        body: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "file_id": chunk.file_id,
            "content": content if content is not None else chunk.content,
            "order": chunk.order,
            "vector": vector,
        }
        
        # Add metadata - either as nested object or flattened based on config
        if self.chunk_fields_config.get("flatten_metadata", False):
            # Flatten metadata fields into top-level document
            for field_name in configured_fields:
                if field_name in chunk.metadata:
                    body[field_name] = chunk.metadata[field_name]
        else:
            # Store metadata as nested object (default behavior)
            body["metadata"] = chunk.metadata
        
        # Add chunk-level fields if they exist
        if chunk.parent_id and "parent_id" in configured_fields:
            body["parent_id"] = chunk.parent_id
        if chunk.level and "level" in configured_fields:
            body["level"] = chunk.level
        
        return body

    def verify_chunk_length(self, chunk: Chunk) -> ChunkValidationResult:
        """
        Verify that a chunk's content length is within acceptable bounds.
        
        Args:
            chunk: The chunk to validate
            
        Returns:
            ChunkValidationResult with validation status and details
        """
        content_length = len(chunk.content) if chunk.content else 0
        
        if content_length < self.min_chunk_length:
            return ChunkValidationResult(
                is_valid=False,
                chunk_id=chunk.chunk_id,
                content_length=content_length,
                reason=f"Chunk content too short: {content_length} chars (min: {self.min_chunk_length})"
            )
        
        if content_length > self.max_chunk_length:
            return ChunkValidationResult(
                is_valid=False,
                chunk_id=chunk.chunk_id,
                content_length=content_length,
                reason=f"Chunk content too long: {content_length} chars (max: {self.max_chunk_length})"
            )
        
        return ChunkValidationResult(
            is_valid=True,
            chunk_id=chunk.chunk_id,
            content_length=content_length
        )

    def verify_all_chunks(
        self, 
        embeddings: List[Tuple[Chunk, List[float]]]
    ) -> Tuple[List[ChunkValidationResult], List[ChunkValidationResult], List[ChunkValidationResult]]:
        """
        Verify all chunks and categorize them by validation status.
        
        Args:
            embeddings: List of (chunk, vector) tuples to validate
            
        Returns:
            Tuple of (valid_results, invalid_results, warning_results)
        """
        valid_results: List[ChunkValidationResult] = []
        invalid_results: List[ChunkValidationResult] = []
        warning_results: List[ChunkValidationResult] = []
        
        for chunk, _ in embeddings:
            result = self.verify_chunk_length(chunk)
            
            if result.is_valid:
                # Check for warning threshold
                if result.content_length > self.warn_chunk_length:
                    result.reason = f"Chunk content approaching limit: {result.content_length} chars (warn threshold: {self.warn_chunk_length})"
                    warning_results.append(result)
                valid_results.append(result)
            else:
                invalid_results.append(result)
        
        return valid_results, invalid_results, warning_results

    def upsert(self, embeddings: List[Tuple[Chunk, List[float]]], index_name: Optional[str] = None) -> None:
        target_index = index_name or self.index
        
        # Perform chunk length validation if enabled
        if self.validate_chunk_length:
            valid_results, invalid_results, warning_results = self.verify_all_chunks(embeddings)
            
            # Log validation summary
            LOGGER.log_chunk_validation_summary(
                total_chunks=len(embeddings),
                valid_count=len(valid_results),
                invalid_count=len(invalid_results),
                warning_count=len(warning_results)
            )
            
            # Log warnings for chunks approaching limits
            for warning in warning_results:
                LOGGER.log_chunk_validation_warning(
                    chunk_id=warning.chunk_id,
                    content_length=warning.content_length,
                    reason=warning.reason
                )
            
            # Handle invalid chunks
            for invalid in invalid_results:
                LOGGER.log_chunk_validation_error(
                    chunk_id=invalid.chunk_id,
                    content_length=invalid.content_length,
                    reason=invalid.reason
                )
                
                if self.skip_invalid_chunks:
                    LOGGER.log_debug(
                        ProcessLogger.STAGE_INDEX, 
                        f"Skipping invalid chunk: {invalid.chunk_id}"
                    )
            
            # Filter out invalid chunks if configured to skip them
            if self.skip_invalid_chunks and invalid_results:
                invalid_chunk_ids = {r.chunk_id for r in invalid_results}
                embeddings = [
                    (chunk, vector) for chunk, vector in embeddings 
                    if chunk.chunk_id not in invalid_chunk_ids
                ]
        
        # Build all document bodies for bulk indexing
        documents = []
        for chunk, vector in embeddings:
            content = chunk.content
            
            # Truncate content if configured and exceeds max length
            if self.truncate_long_chunks and len(content) > self.max_chunk_length:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_INDEX,
                    f"Truncating chunk {chunk.chunk_id} from {len(content)} to {self.max_chunk_length} chars"
                )
                content = content[:self.max_chunk_length]
            
            # Build document body using configured fields
            body = self.build_document_body(chunk, vector, content=content)
            
            # Add document ID if custom IDs are allowed
            if self.allow_custom_ids and chunk.chunk_id:
                body["_doc_id"] = chunk.chunk_id
            
            documents.append(body)
        
        if not documents:
            LOGGER.log_debug(ProcessLogger.STAGE_INDEX, "No documents to index")
            return
        
        # Use bulk indexing if available, fall back to single-document indexing
        if hasattr(self.vectordb, 'bulk_index_documents'):
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Bulk indexing {len(documents)} chunks to {target_index} "
                f"(chunk_size={self.bulk_chunk_size}, timeout={self.bulk_request_timeout}s)"
            )
            
            # Strip _doc_id from document bodies before bulk indexing.
            # The cloud_lib bulk adapter copies id_field to the provider action
            # but does not remove it from the stored document. Some vector
            # database providers reject fields with reserved prefixes.
            # Use 'chunk_id' (already in every doc body) as id_field instead.
            for doc in documents:
                doc.pop("_doc_id", None)
            
            bulk_id_field = "chunk_id" if self.allow_custom_ids else None
            
            result = self.vectordb.bulk_index_documents(
                index_name=target_index,
                documents=documents,
                id_field=bulk_id_field,
                chunk_size=self.bulk_chunk_size,
                request_timeout=self.bulk_request_timeout,
            )
            
            success_count = result.get('success_count', 0)
            error_count = result.get('error_count', 0)
            errors = result.get('errors', [])
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Bulk index complete: {success_count} succeeded, {error_count} failed"
            )
            
            if errors:
                for error in errors[:5]:  # Log first 5 errors
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_INDEX,
                        f"Bulk index error: {error}"
                    )
                if len(errors) > 5:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_INDEX,
                        f"... and {len(errors) - 5} more errors"
                    )
            
            # Raise exception on ANY indexing errors (not just total failure)
            if error_count > 0:
                error_details = "; ".join(str(e) for e in errors[:3]) if errors else "unknown"
                raise CloudlibException(
                    f"Vector database bulk indexing failed: {error_count}/{len(documents)} documents failed. "
                    f"Errors: {error_details}",
                    operation="bulk_index_documents",
                    service="vectordb",
                    category="service_error",
                )
        else:
            # Fallback: single-document indexing for adapters without bulk support
            LOGGER.log_stage(
                ProcessLogger.STAGE_INDEX,
                f"Indexing {len(documents)} chunks individually to {target_index}"
            )
            for body in documents:
                doc_id = body.pop("_doc_id", None)
                assigned_id = self.vectordb.index_document(target_index, body, document_id=doc_id)
                if assigned_id:
                    LOGGER.log_debug(ProcessLogger.STAGE_INDEX, f"Document stored as: {assigned_id}")

    def ensure_index(
        self,
        vector_dimension: Optional[int],
        body: Optional[Dict[str, Any]] = None,
        *,
        knn_method: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> None:
        """Create the backing index with a sensible mapping if it is missing."""
        target_index = index_name or self.index
        if not body and vector_dimension is None:
            raise ValueError("vector_dimension required when auto-creating VectorDB index")

        created = self.vectordb.ensure_vector_index(
            index_name=target_index,
            vector_dimension=vector_dimension,
            index_body=body,
            knn_method=knn_method,
            field_config=self.chunk_fields_config,
        )
        action = "Created" if created else "Found"
        LOGGER.log_stage(ProcessLogger.STAGE_INDEX, f"{action} VectorDB index: {target_index}")

    def ensure_index_from_config(
        self,
        vectordb_cfg: Dict[str, Any],
        index_name: Optional[str] = None,
    ) -> None:
        """
        Ensure vector index exists based on configuration.
        
        Args:
            vectordb_cfg: VectorDB configuration dictionary containing:
                - auto_create_index: bool (default True)
                - vector_dimension: int
                - index_body: dict (optional)
                - knn_method: dict (optional)
                - enforce_vector_dimension: bool (optional)
            index_name: Optional override for index name
        """
        if not vectordb_cfg:
            LOGGER.log_warning(ProcessLogger.STAGE_INIT, "VectorDB configuration missing; skipping index bootstrap")
            return

        target_index = index_name or self.index
        if not vectordb_cfg.get("auto_create_index", True):
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Auto index creation disabled; assuming index '{target_index}' exists")
            return

        try:
            self.ensure_index(
                vector_dimension=vectordb_cfg.get("vector_dimension"),
                body=vectordb_cfg.get("index_body"),
                knn_method=vectordb_cfg.get("knn_method"),
                index_name=target_index
            )
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"VectorDB index '{target_index}' ensured")
        except Exception as e:
            LOGGER.log_error(f"Failed to ensure VectorDB index '{target_index}' exists", exception=e)
            raise

    def _execute_metadata_delete(
        self,
        matches: Dict[str, Any],
        exclusions: Optional[Dict[str, Any]],
        target_index: str,
        batch_size: int,
        log_context: str,
    ) -> int:
        """Delete selected documents through the semantic metadata contract."""
        try:
            result = self.vectordb.delete_documents_by_metadata(
                index_name=target_index,
                matches=matches,
                exclusions=exclusions,
                request_timeout=self.bulk_request_timeout,
                batch_size=batch_size,
            )
            deleted = result.get('deleted_count', 0)
            real_errors = result.get('errors', []) or []
            benign_count = result.get('ignored_error_count', 0)

            if benign_count:
                LOGGER.log_debug(
                    ProcessLogger.STAGE_DELETE,
                    f"{log_context}: {benign_count} benign delete errors ignored"
                )

            if real_errors:
                for error in real_errors[:5]:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_DELETE,
                        f"{log_context}: delete error: {error}"
                    )
                if len(real_errors) > 5:
                    LOGGER.log_warning(
                        ProcessLogger.STAGE_DELETE,
                        f"{log_context}: ... and {len(real_errors) - 5} more delete errors"
                    )

            LOGGER.log_debug(
                ProcessLogger.STAGE_DELETE,
                f"{log_context}: deleted {deleted} documents from {target_index}"
            )
            return deleted

        except Exception as e:
            error_msg = str(e).lower()
            if "index_not_found_exception" in error_msg or "no such index" in error_msg:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_DELETE,
                    f"{log_context}: index {target_index} not found; assuming empty"
                )
                return 0
            raise

    def delete_by_file_id(self, file_id: str, batch_size: int = 200, index_name: Optional[str] = None) -> int:
        """
        Delete all chunks associated with a file_id.

        Uses cloud_lib's delete_by_query implementation, which selects the
        provider-supported native or search-and-bulk-delete strategy.
        """
        if not file_id:
            raise ValueError("file_id is required")
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("batch_size must be between 1 and 1000")

        target_index = index_name or self.index
        return self._execute_metadata_delete(
            matches={"file_id": file_id},
            exclusions=None,
            target_index=target_index,
            batch_size=batch_size,
            log_context=f"delete_by_file_id(file_id={file_id})",
        )

    def delete_stale_chunks_for_file(
        self,
        file_id: str,
        current_ingestion_run_id: str,
        batch_size: int = 200,
        index_name: Optional[str] = None,
    ) -> int:
        """
        Delete chunks for ``file_id`` that do NOT belong to the current
        ingestion run.

        This is the "sweep" half of the **index-then-sweep** re-ingestion
        strategy:

        1. New chunks are indexed first, each tagged with
           ``metadata.ingestion_run_id = <current run uuid>``.
        2. After successful indexing, this method removes every chunk for the
           same ``file_id`` whose ``ingestion_run_id`` is anything else (older
           runs) **or** whose ``ingestion_run_id`` field is missing entirely
           (legacy chunks indexed before this strategy was introduced).

        Failure modes are deliberately *non-fatal* for the caller — if the
        sweep cannot complete (for example, after a transient vector database error), the new chunks are
        already live and a future re-ingestion will clean up any leftovers.

        Args:
            file_id: The file ID whose stale chunks should be removed.
            current_ingestion_run_id: The run-id stamped on the just-indexed
                new chunks; chunks with this exact value will be preserved.
            batch_size: Number of documents per batch.
            index_name: Optional override for target index.

        Returns:
            Number of stale chunks deleted.
        """
        if not file_id:
            raise ValueError("file_id is required")
        if not current_ingestion_run_id:
            raise ValueError("current_ingestion_run_id is required")
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("batch_size must be between 1 and 1000")

        target_index = index_name or self.index

        return self._execute_metadata_delete(
            matches={"file_id": file_id},
            exclusions={"ingestion_run_id": current_ingestion_run_id},
            target_index=target_index,
            batch_size=batch_size,
            log_context=(
                f"sweep_stale(file_id={file_id}, "
                f"keep_run_id={current_ingestion_run_id})"
            ),
        )
