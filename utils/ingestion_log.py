"""Flush accumulated ingestion events to cloud object storage as JSONL."""
from __future__ import annotations

import json
import logging
import posixpath
from io import BytesIO
from typing import Optional

from .logger import get_ingestion_events, clear_ingestion_events, get_logger, ProcessLogger

LOGGER = get_logger(__name__)


def flush_ingestion_log(
    file_manager,
    bucket: str,
    document_prefix: Optional[str],
    file_id: str,
    base_prefix: str = "",
) -> Optional[str]:
    """Flush accumulated ingestion log events to cloud object storage as a JSONL file.

    Args:
        file_manager: CloudFactory file manager instance (has upload_file method).
        bucket: Cloud object storage container name.
        document_prefix: Object storage key prefix for the document directory (e.g.
            ``knowledgebases/kb/sgt/sgi/local/file_id``).
            If None, falls back to ``{base_prefix}/{file_id}/``.
        file_id: Document file_id.
        base_prefix: Fallback prefix when document_prefix is None.

    Returns:
        The object storage URI of the stored log, or None if no events to flush.
    """
    events = get_ingestion_events()
    if not events:
        clear_ingestion_events()
        return None

    # Build object storage key
    if document_prefix:
        key = posixpath.join(document_prefix, "ingestion_log.jsonl").strip("/")
    else:
        key = posixpath.join(base_prefix, file_id, "ingestion_log.jsonl").strip("/")

    # Serialize events as newline-delimited JSON
    lines = [json.dumps(evt, default=str) for evt in events]
    content = "\n".join(lines).encode("utf-8")

    try:
        file_manager.upload_file(
            BytesIO(content),
            bucket,
            key,
            ContentType="application/x-ndjson",
        )
        location = file_manager.build_uri(bucket, key)
        LOGGER.log_stage(
            ProcessLogger.STAGE_STORAGE,
            f"Ingestion log stored: {location} ({len(events)} events)",
        )
        return location
    except Exception as e:
        # Non-fatal — don't let log persistence break the pipeline
        LOGGER.log_warning(
            ProcessLogger.STAGE_STORAGE,
            f"Failed to store ingestion log at {bucket}/{key}: {e}",
        )
        return None
    finally:
        clear_ingestion_events()
