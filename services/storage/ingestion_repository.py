"""Ingestion Repository - Application-level database operations for document ingestion.

This module contains all ingestion-specific database logic, separating business
logic from the generic cloud database adapters.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from cloud_lib.interfaces.database_manager import DatabaseManager
from utils.logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class IngestionRepository:
    """Repository for managing document ingestion records in the database.
    
    This class encapsulates all database operations related to tracking
    document ingestion status, strategies, and results. It uses the generic
    DatabaseManager interface, making it cloud-agnostic.
    
    Args:
        db_manager: A cloud_lib DatabaseManager instance
        schema: PostgreSQL schema name (defaults to 'public')
    """
    
    # Ingestion status constants
    STATUS_PENDING = "PENDING"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_COMPLETED = "COMPLETED"
    STATUS_FAILED = "FAILED"
    STATUS_DELETED = "DELETED"
    
    # Status mapping from tbl_data_ingestion_files to tbl_files.ingestion_status_enum
    # tbl_files enum only has: pending, completed, failed (no 'processing')
    _TBL_FILES_STATUS_MAP = {
        "COMPLETED": "completed",
        "FAILED": "failed",
    }
    
    def __init__(self, db_manager: DatabaseManager, schema: str = "public"):
        self.db = db_manager
        self.schema = schema
        self.TABLE_NAME = f"{schema}.tbl_data_ingestion_files"
        self.TBL_FILES = f"{schema}.tbl_files"
    
    def update_status(
        self,
        file_id: str,
        status: str,
        strategies: Optional[Dict[str, str]] = None,
        error_message: Optional[str] = None,
        index_name: Optional[str] = None,
        chunk_id_list: Optional[List[str]] = None,
        ingestion_duration_minutes: Optional[float] = None,
        markdown_location: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> None:
        """Update the ingestion status for a file in the database.
        
        Uses UPSERT (INSERT ... ON CONFLICT UPDATE) to create or update records.
        Only non-null values will overwrite existing values (COALESCE pattern).
        
        Args:
            file_id: Unique identifier for the file
            status: Ingestion status (PENDING, PROCESSING, COMPLETED, FAILED, DELETED)
            strategies: Dictionary of processing strategies used:
                       - markdown_strategy
                       - chunking_strategy  
                       - embedding_strategy
                       - image_description_strategy
            error_message: Error message if ingestion failed
            index_name: Name of the vector database index
            chunk_id_list: List of chunk IDs created during processing
            ingestion_duration_minutes: Time taken to ingest document in minutes
            markdown_location: Storage URI where the converted markdown file is stored
            file_type: File type (html, pdf, docx, xlsx, pptx, txt, etc.)
        """
        strategies = strategies or {}
        chunk_ids_json = json.dumps(chunk_id_list) if chunk_id_list else None
        # file_type can come from strategies dict or explicit parameter
        resolved_file_type = file_type or strategies.get('file_type')
        
        query = f"""
            INSERT INTO {self.TABLE_NAME} (
                file_id, ingestion_status, ingestion_failure, updated_at,
                markdown_strategy, chunking_strategy, embedding_strategy, 
                image_description_strategy, index_name, chunk_id_list,
                ingestion_duration_minutes, markdown_location, file_type
            )
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_id) 
            DO UPDATE SET 
                ingestion_status = EXCLUDED.ingestion_status,
                ingestion_failure = EXCLUDED.ingestion_failure,
                updated_at = CURRENT_TIMESTAMP,
                markdown_strategy = COALESCE(EXCLUDED.markdown_strategy, {self.TABLE_NAME}.markdown_strategy),
                chunking_strategy = COALESCE(EXCLUDED.chunking_strategy, {self.TABLE_NAME}.chunking_strategy),
                embedding_strategy = COALESCE(EXCLUDED.embedding_strategy, {self.TABLE_NAME}.embedding_strategy),
                image_description_strategy = COALESCE(EXCLUDED.image_description_strategy, {self.TABLE_NAME}.image_description_strategy),
                index_name = COALESCE(EXCLUDED.index_name, {self.TABLE_NAME}.index_name),
                chunk_id_list = COALESCE(EXCLUDED.chunk_id_list, {self.TABLE_NAME}.chunk_id_list),
                ingestion_duration_minutes = COALESCE(EXCLUDED.ingestion_duration_minutes, {self.TABLE_NAME}.ingestion_duration_minutes),
                markdown_location = COALESCE(EXCLUDED.markdown_location, {self.TABLE_NAME}.markdown_location),
                file_type = COALESCE(EXCLUDED.file_type, {self.TABLE_NAME}.file_type);
        """
        
        params = (
            file_id, 
            status, 
            error_message,
            strategies.get('markdown_strategy'),
            strategies.get('chunking_strategy'),
            strategies.get('embedding_strategy'),
            strategies.get('image_description_strategy'),
            index_name,
            chunk_ids_json,
            ingestion_duration_minutes,
            markdown_location,
            resolved_file_type
        )
        
        try:
            self.db.execute(query, params)
            LOGGER.log_stage(
                ProcessLogger.STAGE_DB_UPDATE,
                f"Updated ingestion status: {status}",
                extra_data={"status": status, "index_name": index_name},
            )
            # Sync status to tbl_files for KB Builder compatibility
            self._sync_tbl_files_status(file_id, status, index_name)
        except Exception as e:
            LOGGER.log_error(
                f"Failed to update ingestion status: {status}",
                exception=e,
                extra_data={"status": status},
            )
            raise
    
    def get_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get the ingestion status for a file from the database.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            Dictionary with all ingestion record fields, or None if not found
        """
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE file_id = %s"
        
        try:
            result = self.db.fetch_one(query, (file_id,))
            return result
        except Exception as e:
            LOGGER.log_error(
                f"Failed to get ingestion status",
                exception=e,
            )
            raise
    
    def _sync_tbl_files_status(
        self,
        file_id: str,
        status: str,
        index_name: Optional[str] = None,
    ) -> None:
        """Sync ingestion status to tbl_files for KB Builder compatibility.
        
        The KB Builder reads tbl_files.ingestion_status to track document
        processing progress. This keeps it in sync with tbl_data_ingestion_files.
        Only COMPLETED and FAILED are synced (tbl_files enum has no 'processing').
        
        Args:
            file_id: Unique identifier for the file
            status: The status from tbl_data_ingestion_files (uppercase)
            index_name: Target index name to append to index_names
        """
        mapped = self._TBL_FILES_STATUS_MAP.get(status)
        if not mapped:
            return
        
        try:
            update_query = f"""
                UPDATE {self.TBL_FILES}
                SET ingestion_status = %s::{self.schema}.ingestion_status_enum,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_id = %s
            """
            self.db.execute(update_query, (mapped, file_id))
            
            # Also update index_names if an index was used
            if index_name and mapped == "completed":
                index_query = f"""
                    UPDATE {self.TBL_FILES}
                        SET index_names = CASE
                            WHEN NOT index_names @> %s::jsonb
                            THEN index_names || %s::jsonb
                            ELSE index_names
                        END
                    WHERE file_id = %s
                """
                index_json = json.dumps([index_name])
                self.db.execute(index_query, (index_json, index_json, file_id))
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_DB_UPDATE,
                f"Synced tbl_files status: {mapped}",
                extra_data={"file_id": file_id, "status": mapped},
            )
        except Exception as e:
            # Non-fatal: log warning but don't fail the pipeline
            LOGGER.log_error(
                f"Failed to sync tbl_files status (non-fatal)",
                exception=e,
                extra_data={"file_id": file_id, "status": mapped},
            )
    
    def mark_processing(
        self, 
        file_id: str, 
        strategies: Dict[str, str],
        index_name: Optional[str] = None
    ) -> None:
        """Mark a file as currently being processed.
        
        Convenience method for setting PROCESSING status with strategies.
        
        Args:
            file_id: Unique identifier for the file
            strategies: Dictionary of processing strategies being used
            index_name: Target vector database index
        """
        self.update_status(
            file_id=file_id,
            status=self.STATUS_PROCESSING,
            strategies=strategies,
            index_name=index_name
        )
    
    def mark_completed(
        self,
        file_id: str,
        index_name: str,
        chunk_id_list: List[str],
        ingestion_duration_minutes: float,
        markdown_location: Optional[str] = None
    ) -> None:
        """Mark a file as successfully completed.
        
        Convenience method for setting COMPLETED status with results.
        
        Args:
            file_id: Unique identifier for the file
            index_name: Vector database index where chunks were stored
            chunk_id_list: List of created chunk IDs
            ingestion_duration_minutes: Total processing time in minutes
            markdown_location: Storage URI of the converted markdown
        """
        self.update_status(
            file_id=file_id,
            status=self.STATUS_COMPLETED,
            index_name=index_name,
            chunk_id_list=chunk_id_list,
            ingestion_duration_minutes=ingestion_duration_minutes,
            markdown_location=markdown_location
        )
    
    def mark_failed(
        self,
        file_id: str,
        error_message: str,
        index_name: Optional[str] = None,
        ingestion_duration_minutes: Optional[float] = None
    ) -> None:
        """Mark a file as failed.
        
        Convenience method for setting FAILED status with error details.
        
        Args:
            file_id: Unique identifier for the file
            error_message: Description of what went wrong
            index_name: Target vector database index (if known)
            ingestion_duration_minutes: Time spent before failure
        """
        self.update_status(
            file_id=file_id,
            status=self.STATUS_FAILED,
            error_message=error_message,
            index_name=index_name,
            ingestion_duration_minutes=ingestion_duration_minutes
        )
    
    def mark_deleted(self, file_id: str, index_name: Optional[str] = None) -> None:
        """Mark a file as deleted from the vector index.
        
        Args:
            file_id: Unique identifier for the file
            index_name: Vector database index from which it was deleted
        """
        self.update_status(
            file_id=file_id,
            status=self.STATUS_DELETED,
            index_name=index_name
        )
    
    def delete_record(self, file_id: str) -> bool:
        """Permanently delete an ingestion record from the database.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            True if a record was deleted, False if not found
        """
        query = f"DELETE FROM {self.TABLE_NAME} WHERE file_id = %s"
        
        try:
            rows_affected = self.db.execute(query, (file_id,))
            if rows_affected > 0:
                LOGGER.log_stage(
                    ProcessLogger.STAGE_DB_UPDATE,
                    f"Deleted ingestion record",
                )
            return rows_affected > 0
        except Exception as e:
            LOGGER.log_error(
                f"Failed to delete ingestion record",
                exception=e,
            )
            raise
    
    def get_chunk_ids(self, file_id: str) -> Optional[List[str]]:
        """Get the list of chunk IDs for a file.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            List of chunk IDs, or None if not found
        """
        record = self.get_status(file_id)
        if record and record.get('chunk_id_list'):
            chunk_ids = record['chunk_id_list']
            # Handle both JSON string and already-parsed list
            if isinstance(chunk_ids, str):
                return json.loads(chunk_ids)
            return chunk_ids
        return None
