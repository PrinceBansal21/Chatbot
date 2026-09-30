"""Result building utilities for document processing pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .logger import get_logger

LOGGER = get_logger(__name__)


class ResultBuilder:
    """
    Builds standardized result dictionaries for document processing operations.
    
    Provides consistent result formatting for both single-index and multi-index operations.
    """
    
    @staticmethod
    def build_delete_result(
        file_id: str,
        target_indices: List[str],
        deleted_per_index: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Build result dictionary for delete operations.
        
        Args:
            file_id: Document file ID
            target_indices: List of target index names
            deleted_per_index: Number of chunks deleted per index
            
        Returns:
            Result dictionary with deletion details
        """
        total_deleted = sum(deleted_per_index.values())
        
        return {
            "file_id": file_id,
            "operation": "delete",
            "chunks_deleted": total_deleted,
            "deleted_per_index": deleted_per_index,
            "index_names": target_indices,
            # Backward compatibility: include single index_name if only one
            "index_name": target_indices[0] if len(target_indices) == 1 else target_indices,
        }
    
    @staticmethod
    def build_index_result(
        file_id: str,
        deleted: int,
        chunks: list,
        target_index: str,
        strategies: Dict[str, str],
        image_manifest: Optional[Dict[str, Any]] = None,
        markdown_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build result dictionary for single-index operations (backward compatibility).
        
        Args:
            file_id: Document file ID
            deleted: Number of chunks deleted
            chunks: List of chunks created
            target_index: Target index name
            strategies: Processing strategies used
            image_manifest: Image extraction manifest (if any)
            markdown_location: Location of stored markdown
            
        Returns:
            Result dictionary with indexing details
        """
        # Count text and image chunks separately
        text_chunks = [c for c in chunks if c.metadata.get("chunk_type") != "image"]
        image_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image"]
        
        result = {
            "file_id": file_id,
            "operation": "index",
            "chunks_deleted": deleted,
            "chunks_created": len(chunks),
            "text_chunks_created": len(text_chunks),
            "image_chunks_created": len(image_chunks),
            "index_name": target_index,
            "configuration": strategies,
        }
        
        if markdown_location:
            result["markdown_location"] = markdown_location
        
        ResultBuilder._add_image_info(result, image_manifest)
        
        LOGGER.end_processing(file_id, result)
        return result
    
    @staticmethod
    def build_multi_index_result(
        file_id: str,
        deleted_per_index: Dict[str, int],
        chunks: list,
        target_indices: List[str],
        indexed_per_index: Dict[str, int],
        strategies: Dict[str, str],
        image_manifest: Optional[Dict[str, Any]] = None,
        image_chunks_count: int = 0,
        infographic_chunks_count: int = 0,
        markdown_location: Optional[str] = None,
        graph_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build result dictionary for multi-index operations.
        
        Args:
            file_id: Document file ID
            deleted_per_index: Chunks deleted per index
            chunks: List of text chunks created
            target_indices: List of target index names
            indexed_per_index: Total chunks indexed per index (text + image + infographic)
            strategies: Processing strategies used
            image_manifest: Image extraction manifest (if any)
            image_chunks_count: Number of image chunks created per index
            infographic_chunks_count: Number of infographic chunks created per index
            markdown_location: Location of stored markdown
            graph_result: Graph knowledge base extraction results (if any)
            
        Returns:
            Result dictionary with multi-index details
        """
        # Count text and image chunks separately
        text_chunks = [c for c in chunks if c.metadata.get("chunk_type") != "image"]
        
        total_deleted = sum(deleted_per_index.values())
        total_new_chunks = len(text_chunks) + image_chunks_count + infographic_chunks_count
        
        result = {
            "file_id": file_id,
            "operation": "index",
            "chunks_deleted": total_deleted,
            "deleted_per_index": deleted_per_index,
            "chunks_created": total_new_chunks,
            "text_chunks_created": len(text_chunks),
            "image_chunks_created": image_chunks_count,
            "infographic_chunks_created": infographic_chunks_count,
            "indexed_per_index": indexed_per_index,
            "index_names": target_indices,
            # Backward compatibility: include single index_name if only one
            "index_name": target_indices[0] if len(target_indices) == 1 else target_indices,
            "configuration": strategies,
        }
        
        if markdown_location:
            result["markdown_location"] = markdown_location
        
        if graph_result:
            result["graph_result"] = graph_result
        
        ResultBuilder._add_image_info(result, image_manifest)
        
        LOGGER.end_processing(file_id, result)
        return result
    
    @staticmethod
    def build_error_result(error_message: str) -> Dict[str, Any]:
        """
        Build result dictionary for error cases.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Result dictionary with error information
        """
        return {"error": error_message}
    
    @staticmethod
    def _add_image_info(
        result: Dict[str, Any],
        image_manifest: Optional[Dict[str, Any]],
    ) -> None:
        """
        Add image information to result dictionary.
        
        Args:
            result: Result dictionary to update (modified in place)
            image_manifest: Image extraction manifest
        """
        if image_manifest:
            result["image_count"] = image_manifest.get("image_count", 0)
            result["image_manifest_location"] = image_manifest.get("manifest_location")
            result["has_images"] = image_manifest.get("image_count", 0) > 0


__all__ = ["ResultBuilder"]
