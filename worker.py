"""Cloud-agnostic worker for processing messages from a queue."""
import logging
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from app import Pipeline as IngestionPipeline
from cloud_lib.factory import CloudFactory
from utils.logger import configure_root_logger, get_logger, set_file_id, clear_file_id, ProcessLogger

# Configure structured logging
configure_root_logger(level=logging.INFO)
LOGGER = get_logger(__name__)


class CloudWorker:
    """
    Cloud-agnostic worker that polls messages from a queue and processes them.
    
    Uses the queue manager selected by CloudFactory.
    """
    
    def __init__(self):
        """Initialize the cloud worker with queue manager and pipeline."""
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Starting Cloud Worker")
        
        # Get Queue Name from environment or deployment dependencies
        self.queue_name = self._get_queue_name()
        if not self.queue_name:
            LOGGER.log_error("Queue name not configured. Set QUEUE_NAME or configure queues.ingestion")
            sys.exit(1)
        self.queue_manager = CloudFactory.get_queue_manager(queue_name=self.queue_name)
        
        # Log resolved queue details for diagnosability
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            f"Queue resolved: name/url='{self.queue_name}'",
            extra_data={"queue_name": self.queue_name},
        )
        
        # Initialize Pipeline
        try:
            self.pipeline = IngestionPipeline()
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "IngestionPipeline initialized successfully")
        except Exception as e:
            import traceback
            LOGGER.log_error(
                f"Failed to initialize IngestionPipeline: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                exception=e,
            )
            sys.exit(1)
        
        # Configure worker pool
        self.max_workers = self._get_max_workers()
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT, 
            f"Cloud Worker ready",
            extra_data={
                "max_workers": self.max_workers, 
                "queue": self.queue_name
            }
        )

    def _get_queue_name(self) -> Optional[str]:
        """Get queue name from environment variables or deployment dependencies.
        
        Resolution order:
        1. QUEUE_URL / QUEUE_NAME env vars (generic)
        2. Pipeline deployment dependencies (from merged config via cloud config store)
        3. Legacy infrastructure config fallback
        """
        # Generic env vars
        queue_name = os.environ.get('QUEUE_URL') or os.environ.get('QUEUE_NAME')
        if queue_name:
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Queue from env (QUEUE_URL/QUEUE_NAME): {queue_name}")
            return queue_name
        
        # Try deployment dependencies (same config the pipeline uses)
        queue_name = self._get_queue_from_deployment_dependencies()
        if queue_name:
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Queue from deployment dependencies: {queue_name}")
            return queue_name
        
        return None

    def _get_queue_from_deployment_dependencies(self) -> Optional[str]:
        """Load queue name from the merged pipeline configuration."""
        try:
            from services.config import ConfigLoader
            from models import DeploymentDependencies
            loader = ConfigLoader()
            config = loader.load()
            dependencies = DeploymentDependencies.from_dict(config.get("deployment_dependencies"))
            queue_config = config.get("queues", {})
            return dependencies.resolved_queue("ingestion", queue_config.get("ingestion"))
        except Exception as e:
            LOGGER.log_warning(ProcessLogger.STAGE_INIT, f"Failed to load queue from deployment dependencies: {e}")
            return None

    def _get_max_workers(self) -> int:
        """Get the required worker concurrency from merged cloud configuration."""
        try:
            config = self.pipeline.config
            worker = config.get('worker', {})
            configured = worker['max_workers']
        except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Missing required worker.max_workers in pipeline cloud configuration."
            ) from exc

        max_workers = int(configured)
        if max_workers < 1:
            raise ValueError("pipeline cloud configuration worker.max_workers must be positive.")
        return max_workers

    def process_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single message.
        Catches all exceptions to ensure one failure does not impact other threads.
        
        Returns:
            A dict with ``status`` ('success' or 'failed'), ``file_id``, and
            on failure ``error`` and ``error_type`` — used by ``run()`` to
            build the batch summary.
        """
        normalized = self.queue_manager.normalize_message(msg)
        msg_id = normalized["id"]
        file_id = None
        
        try:
            body = normalized["body"]
            file_id = body.get('file_id', 'unknown')
            set_file_id(file_id)
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_INPUT, 
                f"Processing message",
                extra_data={"message_id": msg_id, "queue": self.queue_name}
            )
            
            self.pipeline.process(body)
            
            # Delete/acknowledge message after successful processing
            self.queue_manager.delete_message(normalized["receipt_handle"])
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_COMPLETE, 
                f"Successfully processed and acknowledged message",
                extra_data={"message_id": msg_id}
            )
            return {"status": "success", "file_id": file_id, "message_id": msg_id}
            
        except Exception as e:
            LOGGER.log_error(
                f"Failed to process message. It will return to queue after visibility timeout.",
                exception=e,
                extra_data={"message_id": msg_id}
            )
            return {
                "status": "failed",
                "file_id": file_id or "unknown",
                "message_id": msg_id,
                "error": str(e)[:500],
                "error_type": type(e).__name__,
            }
        finally:
            clear_file_id()

    def run(self) -> None:
        """
        Main loop: poll for messages and process them using a thread pool.
        
        After each polling batch completes, a **batch summary** is logged
        so that failures are always visible at the end — no document
        extraction information is silently lost.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                try:
                    # Poll for messages
                    messages = self.queue_manager.dequeue(
                        max_messages=self.max_workers,
                        wait_time_seconds=20
                    )

                    if not messages:
                        LOGGER.log_debug(ProcessLogger.STAGE_INPUT, "No messages received, waiting...")
                        continue

                    LOGGER.log_stage(
                        ProcessLogger.STAGE_INPUT, 
                        f"Received {len(messages)} messages from queue"
                    )
                    
                    # Submit tasks to the executor
                    futures = {
                        executor.submit(self.process_message, msg): msg
                        for msg in messages
                    }
                    
                    # Collect results from all futures
                    batch_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            LOGGER.log_error(f"Unexpected error in future result", exception=e)
                            batch_results.append({
                                "status": "failed",
                                "file_id": "unknown",
                                "error": str(e)[:500],
                                "error_type": type(e).__name__,
                            })

                    # Log batch summary (always, so failures are never silent)
                    self._log_batch_summary(batch_results)
                    
                    # Memory cleanup after each batch to prevent memory buildup
                    # in long-running worker processes
                    self.pipeline.cleanup_caches()

                except Exception as e:
                    LOGGER.log_error("Error in main loop", exception=e)
                    # Sleep briefly to avoid tight loop in case of persistent errors
                    time.sleep(5)

    def _log_batch_summary(self, results: list) -> None:
        """Log a summary after each polling batch completes.

        Always emitted so operators can see at a glance how many documents
        succeeded vs. failed.  Failed documents include file_id and error
        type so no extraction information is silently lost.
        """
        if not results:
            return

        succeeded = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]

        if not failed:
            # All succeeded — brief info-level log
            LOGGER.log_stage(
                ProcessLogger.STAGE_COMPLETE,
                f"Batch complete: {len(succeeded)}/{len(results)} succeeded",
            )
            return

        # Build a concise failure listing
        failure_details = []
        for f in failed:
            failure_details.append({
                "file_id": f.get("file_id", "unknown"),
                "error_type": f.get("error_type", "Unknown"),
                "error": f.get("error", "")[:200],
            })

        LOGGER.log_stage(
            ProcessLogger.STAGE_ERROR,
            f"BATCH SUMMARY — {len(succeeded)} succeeded, {len(failed)} FAILED "
            f"out of {len(results)} messages",
            level=logging.ERROR,
            extra_data={
                "total": len(results),
                "succeeded": len(succeeded),
                "failed": len(failed),
                "succeeded_file_ids": [r.get("file_id") for r in succeeded],
                "failure_details": failure_details,
            },
        )

        # Also log each failure on its own line for easy grep
        for f in failed:
            LOGGER.log_stage(
                ProcessLogger.STAGE_ERROR,
                f"FAILED: file_id={f.get('file_id', 'unknown')} "
                f"error_type={f.get('error_type', 'Unknown')} "
                f"error={f.get('error', '')[:200]}",
                level=logging.ERROR,
            )


def main():
    """Entry point for the cloud worker."""
    worker = CloudWorker()
    worker.run()


if __name__ == "__main__":
    main()
