"""Image storage repository for document images."""
from __future__ import annotations

import json
import logging
import os
import posixpath
from io import BytesIO
from typing import Any, Dict, List, Optional

from cloud_lib.factory import CloudFactory

from utils.logger import get_logger, ProcessLogger

LOGGER = get_logger(__name__)

class ImageRepository:
    """
    Persists extracted document images to cloud storage.
    
    Supports multiple cloud providers:
    - Cloud object storage through the selected file manager
    
    Stores images in a file-id-centric structure:
        {bucket}/{document_id}/media/
            ├── {image_id}.png
            ├── {image_id}.jpeg
            └── manifest.json
    """

    def __init__(
        self,
        region: str,
        bucket_uri: str,
        encryption_key: Optional[str] = None,
    ):
        """
        Initialize the image repository.
        
        Args:
            region: Cloud region
            bucket_uri: Provider-native cloud storage container URI
            encryption_key: Optional provider-native encryption key reference,
                resolved into provider-specific upload kwargs via the file manager
        """
        self.encryption_key = encryption_key
        self.region = region
        self.file_manager = CloudFactory.get_file_manager(bucket_key="image_output")
        self.bucket, self.base_prefix = self.file_manager.parse_uri(bucket_uri)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            f"ImageRepository initialized: {self.bucket}/{self.base_prefix}"
        )

    def _get_extra_args(self, content_type: str = "image/png") -> Dict[str, Any]:
        """Get extra arguments for upload (encryption, content type); encryption kwargs are provider-specific and resolved by the file manager."""
        extra_args = {"ContentType": content_type}
        extra_args.update(self.file_manager.encryption_kwargs(self.encryption_key))
        return extra_args

    def _build_storage_uri(self, key: str) -> str:
        """Build a cloud-agnostic storage URI."""
        return self.file_manager.build_uri(self.bucket, key)

    def put_image(
        self,
        document_id: str,
        image_id: str,
        image_bytes: bytes,
        image_format: str = "png",
        document_prefix: Optional[str] = None,
    ) -> str:
        """
        Store a single image to cloud storage.
        
        Args:
            document_id: Unique document identifier
            image_id: Unique image identifier
            image_bytes: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            document_prefix: Optional object storage key prefix where the original document
                lives. When provided, images are stored alongside the document
                as ``{document_prefix}/media/{image_id}.{format}``.
            
        Returns:
            Provider-native cloud storage URI of the uploaded image
        """
        # Build the storage key
        filename = f"{image_id}.{image_format}"
        if document_prefix:
            key = posixpath.join(document_prefix, "media", filename).strip("/")
        else:
            key = posixpath.join(
                self.base_prefix, document_id, "media", filename
            ).strip("/")
        
        # Determine content type
        content_type = f"image/{image_format}"
        if image_format == "jpg":
            content_type = "image/jpeg"
        
        extra_args = self._get_extra_args(content_type)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Uploading image: {self.bucket}/{key}"
        )
        
        file_obj = BytesIO(image_bytes)
        self.file_manager.upload_file(file_obj, self.bucket, key, **extra_args)
        
        return self._build_storage_uri(key)

    def put_images(
        self,
        document_id: str,
        images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Store multiple images to cloud storage.
        
        Args:
            document_id: Unique document identifier
            images: List of image dicts with keys:
                - image_id: str
                - image_bytes: bytes
                - format: str (png, jpeg, etc.)
                
        Returns:
            List of image dicts with added storage_path and storage_url
        """
        stored_images = []
        
        for img in images:
            try:
                storage_uri = self.put_image(
                    document_id=document_id,
                    image_id=img["image_id"],
                    image_bytes=img["image_bytes"],
                    image_format=img.get("format", "png"),
                )
                
                img["storage_path"] = storage_uri
                img["storage_url"] = self._uri_to_url(storage_uri)
                stored_images.append(img)
                
            except Exception as e:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_STORAGE,
                    f"Failed to upload image {img.get('image_id')}: {e}"
                )
                stored_images.append(img)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Stored {len(stored_images)} images for document {document_id}"
        )
        
        return stored_images

    def put_manifest(
        self,
        document_id: str,
        manifest: Dict[str, Any],
        document_prefix: Optional[str] = None,
    ) -> str:
        """
        Store image manifest JSON to cloud storage.
        
        Args:
            document_id: Unique document identifier
            manifest: Manifest dictionary containing image metadata
            document_prefix: Optional object key prefix where the original document
                lives. When provided, the manifest is stored alongside the
                document as ``{document_prefix}/media/manifest.json``.
            
        Returns:
            Storage URI of the manifest file
        """
        if document_prefix:
            key = posixpath.join(document_prefix, "media", "manifest.json").strip("/")
        else:
            key = posixpath.join(
                self.base_prefix, document_id, "media", "manifest.json"
            ).strip("/")
        
        extra_args = self._get_extra_args("application/json")
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Uploading manifest: {self.bucket}/{key}"
        )
        
        manifest_json = json.dumps(manifest, indent=2)
        file_obj = BytesIO(manifest_json.encode("utf-8"))
        self.file_manager.upload_file(file_obj, self.bucket, key, **extra_args)
        
        return self._build_storage_uri(key)

    def _uri_to_url(self, uri: str) -> str:
        """Convert a storage URI to its provider-native permanent URL."""
        return self.file_manager.get_url(uri)

