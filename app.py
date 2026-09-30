"""Cloud-agnostic entrypoint for converting documents and indexing embeddings."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from cloud_lib.factory import CloudFactory
from rag_lib.factory import RAGFactory

# Services - core business logic components
from services import (
    ChunkingStrategyWrapper,
    ConfigLoader,
    DocumentFetcher,
    EmbeddingClient,
    ImageDescriptionAugmenter,
    ImageProcessor,
    ImageRepository,
    IngestionRepository,
    MarkdownConverter,
    MarkdownRepository,
    StrategyResolver,
    SummaryGenerator,
    VectorDBIndexer,
)
from services.document import DocumentProcessor
from services.graph_service import GraphKnowledgeBaseService
from services.indexing import IndexingOrchestrator


# Models - data structures
from models import DeploymentDependencies

# Utils - helper functions, utilities, and logging
from utils import (
    extract_base_metadata,
    resolve_location_label,
    ImageChunkLinker,
    ImageConfigResolver,
    ImageRelevanceScorer,
    MarkdownPreprocessor,
    PayloadParser,
    ResultBuilder,
)
from utils.logger import (
    ProcessLogger,
    configure_root_logger,
    set_file_id,
    clear_file_id,
    get_logger,
    clear_ingestion_events,
)
from utils.ingestion_log import flush_ingestion_log

# Configure structured logging
configure_root_logger(level=logging.INFO)
LOGGER = get_logger(__name__)


class Pipeline:
    """Coordinates the end-to-end document processing."""

    def __init__(self):
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Initializing Pipeline")
        
        # Load configuration
        loader = ConfigLoader()
        self.config = loader.load()
        self.cloud_provider = CloudFactory.get_cloud_provider()
        self.region = self.config.get("region") or os.environ.get("CLOUD_REGION")
        CloudFactory.set_service_name("pipeline")

        # Seed rag_lib model_config cache so adapters can resolve models
        # without needing direct file access (remote config store entries aren't on disk).
        self._seed_model_config()
        
        # Initialize components — each step is wrapped to provide clear error context
        self._safe_init("deployment_dependencies", self._init_deployment_dependencies)
        self._safe_init("document_processing", self._init_document_processing)
        self._safe_init("storage_and_indexing", self._init_storage_and_indexing)
        self._safe_init("optional_services", self._init_optional_services)

        # Compose high-level stage processors
        self._safe_init("document_processor", self._init_document_processor)
        self._safe_init("indexing_orchestrator", self._init_indexing_orchestrator)

        # Validate all lazy services are reachable before processing any documents
        self._safe_init("validate_services", self._validate_services)

        # Config refresh tracking
        self._last_config_refresh = time.time()
        try:
            self._config_refresh_ttl = int(self.config["config_refresh_ttl_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Missing or invalid required config_refresh_ttl_seconds in pipeline cloud configuration."
            ) from exc
        if self._config_refresh_ttl < 1:
            raise ValueError("pipeline cloud configuration config_refresh_ttl_seconds must be positive.")
        
        self._log_dependency_summary()
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Pipeline initialization complete")

    def _safe_init(self, stage_name: str, init_fn) -> None:
        """Run an initialization function with error logging and re-raise."""
        try:
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Starting: {stage_name}")
            init_fn()
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Completed: {stage_name}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            LOGGER.log_error(
                f"Pipeline initialization FAILED at stage '{stage_name}': "
                f"{type(e).__name__}: {e}\n{tb}",
                exception=e,
                extra_data={"stage": stage_name, "error_type": type(e).__name__},
            )
            raise

    # =========================================================================
    # CONFIGURATION MANAGEMENT
    # =========================================================================

    def _seed_model_config(self) -> None:
        """Extract ai-models keys from merged config and seed rag_lib cache.

        The merged config from ConfigLoader contains keys from all split
        config files (processing, ai-models, file-handlers, vectordb-schema).
        We extract only the ai-models portion to seed rag_lib.model_config.
        """
        from rag_lib.model_config import seed_config

        # Keys supplied by the ai-models cloud configuration entry.
        AI_MODELS_KEYS = {
            "model_defaults",
            "vision_llm_models",
            "model_token_limits",
            "inference_profile_required_models",
            "embedding_supported_dimensions",
        }
        ai_models_data = {k: v for k, v in self.config.items() if k in AI_MODELS_KEYS}

        if ai_models_data:
            seed_config(ai_models_data)
            LOGGER.log_stage(
                ProcessLogger.STAGE_INIT,
                f"Seeded model_config cache ({len(ai_models_data)} keys)",
            )

    def refresh_config(self) -> None:
        """Refresh AI model configuration from cloud config store without restarting.

        Call this between document processing runs to pick up changes
        made to the ai-models config parameter. Updates both:
        1. The rag_lib.model_config cache (used by adapters)
        2. The relevant keys in self.config (used by pipeline services)
        3. Re-initializes services that bake model_id/region at init time
        """
        from rag_lib.model_config import refresh_config_from_cloud_config_store

        refreshed = refresh_config_from_cloud_config_store()

        # Also update self.config with refreshed ai-models keys
        AI_MODELS_KEYS = {
            "model_defaults",
            "vision_llm_models",
            "model_token_limits",
            "inference_profile_required_models",
            "embedding_supported_dimensions",
        }
        for k, v in refreshed.items():
            if k in AI_MODELS_KEYS:
                self.config[k] = v

        # Re-initialize services that captured model_id/region at init time
        self._reinit_model_dependent_services()

        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            "AI model config refreshed from cloud store; changes apply to the next document",
        )
        self._last_config_refresh = time.time()

    def _reinit_model_dependent_services(self) -> None:
        """Re-initialize services that store model_id/region in instance vars.

        Called after config refresh so cloud-store changes propagate
        to already-constructed service objects.
        """
        # Summary generator
        try:
            self._init_summary_generator()
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INIT,
                f"Failed to re-init summary_generator after config refresh: {e}",
            )

        # Infographic classifier
        try:
            self._init_infographic_classifier()
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INIT,
                f"Failed to re-init infographic_classifier after config refresh: {e}",
            )

        # Invalidate cached markdown managers (VisionLLMAdapter etc.)
        # so they are recreated with fresh config on next use
        if hasattr(self, "converter") and hasattr(self.converter, "_managers"):
            self.converter._managers.clear()

    def _maybe_refresh_config(self) -> None:
        """Refresh config from the cloud store when its TTL expires.

        Called at the start of each document processing. If the refresh
        fails because of a transient provider error, stale config remains valid
        and processing continues uninterrupted.
        """
        if self._config_refresh_ttl <= 0:
            return  # TTL=0 disables auto-refresh

        elapsed = time.time() - self._last_config_refresh
        if elapsed < self._config_refresh_ttl:
            return

        try:
            self.refresh_config()
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INIT,
                f"Cloud config refresh failed (stale config still valid): {e}",
            )

    # =========================================================================
    # SERVICE VALIDATION
    # =========================================================================

    def _validate_services(self) -> None:
        """Force lazy services to initialize, surfacing credential/config errors at startup.

        This ensures that cloud clients, model adapters, and vector database connections
        are reachable before any documents are processed — failing fast rather
        than mid-batch.
        """
        errors = []

        # 1. Embedding provider
        try:
            self.embedder._get_manager(self.embedder.default_provider)
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Validated embedding provider: {self.embedder.default_provider}")
        except Exception as e:
            errors.append(f"embedding({self.embedder.default_provider}): {e}")

        # 2. Markdown converter
        try:
            self.converter._get_manager(self.converter.default_provider)
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Validated markdown converter: {self.converter.default_provider}")
        except Exception as e:
            errors.append(f"markdown({self.converter.default_provider}): {e}")

        # 3. Chunking strategy
        try:
            self.chunker._get_manager(self.chunker.default_strategy)
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Validated chunking strategy: {self.chunker.default_strategy}")
        except Exception as e:
            errors.append(f"chunking({self.chunker.default_strategy}): {e}")

        # 4. Infographic classifier (Vision LLM)
        if getattr(self, "infographic_classifier", None) is not None:
            try:
                _ = self.infographic_classifier.client
                LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Validated infographic classifier client")
            except Exception as e:
                errors.append(f"infographic_classifier: {e}")

        # 5. Summary generator (LLM Converse API)
        if getattr(self, "summary_generator", None) is not None and self.summary_enabled:
            try:
                _ = self.summary_generator._converse.adapter
                LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Validated summary generator client")
            except Exception as e:
                errors.append(f"summary_generator: {e}")

        # 6. Vision / Image description augmenter
        if getattr(self, "vision", None) is not None:
            try:
                _ = self.vision._get_client(self.vision.default_provider)
                LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Validated vision client")
            except Exception as e:
                errors.append(f"vision({self.vision.default_provider}): {e}")

        # 7. Relational database connection (non-fatal; processing can continue
        #    documents without DB tracking; log details for diagnosability)
        self._validate_database_connection()

        if errors:
            detail = "; ".join(errors)
            raise RuntimeError(f"Service validation failed — {len(errors)} error(s): {detail}")

    def _validate_database_connection(self) -> None:
        """Validate relational database connectivity without blocking startup.

        Attempts a lightweight ``SELECT 1`` to confirm the database is
        reachable and credentials are valid. Logs connection details
        (host, port, dbname, auth method, schema) regardless of outcome.

        This intentionally does NOT append to the ``errors`` list or raise —
        a failed DB connection should not block pipeline initialization since
        document processing can still succeed (ingestion tracking will be
        skipped gracefully).
        """
        db_cfg = self.config.get("database", {})

        if not db_cfg.get("enabled", False):
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Database validation skipped (database disabled)")
            return

        if self.db_manager is None:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INIT,
                "Database validation skipped; db_manager is None "
                "(initialization may have failed earlier)",
                extra_data={"database_config": {
                    "host": db_cfg.get("host"),
                    "port": db_cfg.get("port"),
                    "dbname": db_cfg.get("dbname"),
                }},
            )
            return

        auth_method = (
            getattr(self.db_manager, "auth_mode", "")
            or db_cfg.get("auth_method")
            or "credentials"
        )
        connection_details = {
            "host": db_cfg.get("host"),
            "port": db_cfg.get("port", 5432),
            "dbname": db_cfg.get("dbname"),
            "schema": db_cfg.get("schema", "public"),
            "auth_method": auth_method,
            "username": db_cfg.get("username"),
        }

        try:
            result = self.db_manager.fetch_one("SELECT 1 AS connected")
            LOGGER.log_stage(
                ProcessLogger.STAGE_INIT,
                f"Database connection validated successfully "
                f"(host={db_cfg.get('host')}, db={db_cfg.get('dbname')}, "
                f"auth={auth_method})",
                extra_data=connection_details,
            )
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_INIT,
                f"Database connection validation failed (non-fatal; ingestion tracking "
                f"will be unavailable): {type(e).__name__}: {e}",
                extra_data={**connection_details, "error": str(e), "error_type": type(e).__name__},
            )

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    def _init_deployment_dependencies(self) -> None:
        """Initialize deployment dependencies and queue configuration."""
        self.dependencies = DeploymentDependencies.from_dict(
            self.config.get("deployment_dependencies")
        )
        queue_config = self.config.get("queues", {})
        self.ingestion_queue_name: Optional[str] = self.dependencies.resolved_queue(
            "ingestion", queue_config.get("ingestion")
        )
        self.dead_letter_queue_name: Optional[str] = self.dependencies.resolved_queue("dead_letter")

    def _init_document_processing(self) -> None:
        """Initialize document fetching, conversion, and chunking components."""
        self.fetcher = DocumentFetcher(self.region)
        
        markdown_cfg = self.config.get("markdown_conversion", {})
        self.converter = MarkdownConverter(config=markdown_cfg)
        
        vision_cfg = self.config.get("vision")
        self.vision = ImageDescriptionAugmenter(vision_cfg) if vision_cfg else None
        
        self.embedder = EmbeddingClient(self.config.get("embedding", {}))
        self.chunker = ChunkingStrategyWrapper(self.config.get("chunking", {}), self.embedder)

    def _init_storage_and_indexing(self) -> None:
        """Initialize storage repository and vector database indexer."""
        # Markdown storage
        output_cfg = self.config.get("markdown_storage", {})
        # The encryption key is an opaque provider-native reference.
        encryption_key = self.config.get("storage_encryption", {}).get("key_id")
        
        # Ensure storage bucket exists before creating repositories
        bucket_uri = output_cfg.get("bucket", "")
        self._ensure_storage_container(bucket_uri, "markdown_output")
        
        self.repository = MarkdownRepository(
            region=self.region,
            bucket_uri=output_cfg.get("bucket"),
            encryption_key=encryption_key,
        )
        
        # Image storage and processor
        self._init_image_processing(output_cfg, encryption_key)
        
        # Vector database
        vectordb_cfg = self.config.get("vectordb", {})
        chunk_fields_cfg = self.config.get("chunk_fields", {})
        self.indexer = VectorDBIndexer(
            vectordb_cfg, 
            dependency_index=self.dependencies.vectordb_index,
            chunk_fields_config=chunk_fields_cfg
        )
        self.indexer.ensure_index_from_config(vectordb_cfg)

    def _init_image_processing(self, output_cfg: Dict, encryption_key: Optional[str]) -> None:
        """Initialize image extraction and storage components."""
        image_cfg = self.config.get("image_storage", {})
        image_extraction_cfg = self.config.get("image_extraction", {})
        self.image_extraction_enabled = image_extraction_cfg.get("enabled", False)
        
        if self.image_extraction_enabled:
            image_bucket = image_cfg.get("bucket") or output_cfg.get("bucket")
            self._ensure_storage_container(image_bucket, "image_output")
            self.image_repository = ImageRepository(
                region=self.region,
                bucket_uri=image_bucket,
                encryption_key=encryption_key,
            )
            self.image_processor = ImageProcessor(
                image_repository=self.image_repository,
                config=image_extraction_cfg,
                region=self.region,
            )
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Image extraction enabled")
        else:
            self.image_repository = None
            self.image_processor = None
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Image extraction disabled")

    def _ensure_storage_container(self, storage_uri: str, bucket_key: str) -> None:
        """Ensure the configured cloud storage container is ready."""
        if not storage_uri:
            return

        file_mgr = CloudFactory.get_file_manager(bucket_key=bucket_key)
        container_name = file_mgr.ensure_container(storage_uri, region=self.region)
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            f"Storage container ready: '{container_name}'",
        )

    def _init_optional_services(self) -> None:
        """Initialize optional services: PII Manager, Database Manager, Summary Generator, and Graph KB."""
        self._init_pii_manager()
        self._init_database_manager()
        self._init_strategy_resolver()
        self._init_summary_generator()
        self._init_graph_knowledge_base()
        self._init_infographic_classifier()
        self._init_helper_services()

    def _init_infographic_classifier(self) -> None:
        """Initialize Vision LLM infographic page classifier."""
        from services.llm.infographic_classifier import InfographicClassifier

        infographic_cfg = self.config.get("infographic_classification", {})
        enabled = infographic_cfg.get("enabled", True)

        if not enabled:
            self.infographic_classifier = None
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Infographic classification disabled")
            return

        self.infographic_classifier = InfographicClassifier(
            region=self.region,
            model_id=infographic_cfg.get("model_id"),
            max_workers=infographic_cfg.get("max_workers", 5),
            dpi=infographic_cfg.get("dpi", 150),
        )
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            f"Infographic classifier initialized (model={self.infographic_classifier.model_id})",
        )

    def _init_graph_knowledge_base(self) -> None:
        """Initialize Graph Knowledge Base service if enabled."""
        self.graph_service = GraphKnowledgeBaseService(
            config=self.config,
            region=self.region,
        )
        if self.graph_service.enabled:
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Graph Knowledge Base enabled")
        else:
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Graph Knowledge Base disabled")

    def _init_pii_manager(self) -> None:
        """Initialize PII redaction manager if enabled."""
        self.pii_manager = None
        pii_cfg = self.config.get("pii", {})
        
        if not pii_cfg.get("enabled", False):
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "PII redaction is disabled")
            return
            
        pii_provider = pii_cfg.get("provider", "presidio")
        try:
            self.pii_manager = RAGFactory.get_pii_manager(pii_provider, config=pii_cfg)
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"PII Manager initialized: {pii_provider}")
        except Exception as e:
            LOGGER.log_warning(ProcessLogger.STAGE_INIT, f"Failed to initialize PII Manager: {e}")

    def _init_database_manager(self) -> None:
        """Initialize database tracking through the configured cloud adapter."""
        self.db_manager = None
        self.ingestion_repo = None
        db_cfg = self.config.get("database", {})
        
        if not db_cfg.get("enabled", False):
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Database tracking is disabled")
            return
            
        try:
            self.db_manager = CloudFactory.get_database_manager()
            LOGGER.log_stage(ProcessLogger.STAGE_INIT, f"Database Manager initialized: {db_cfg.get('host')}")
            
            self.ingestion_repo = IngestionRepository(self.db_manager, schema=db_cfg.get("schema", "public"))
        except Exception as e:
            LOGGER.log_error("Failed to initialize Database Manager", exception=e)

    def _init_strategy_resolver(self) -> None:
        """Initialize strategy resolver for processing configuration."""
        self.strategy_resolver = StrategyResolver(
            config=self.config,
            ingestion_repo=self.ingestion_repo,
        )
        # Also keep a reference to the file type resolver for direct access
        self.file_type_resolver = self.strategy_resolver.file_type_resolver

    def _init_summary_generator(self) -> None:
        """Initialize summary generator for document-level summaries."""
        summary_cfg = self.config.get("summary_generation", {})
        # Default to enabled=True for all documents
        if "enabled" not in summary_cfg:
            summary_cfg["enabled"] = True
        
        self.summary_generator = SummaryGenerator(
            config=summary_cfg,
            region=self.region,
        )
        self.summary_enabled = summary_cfg.get("enabled", True)
        self.page_summaries_enabled = summary_cfg.get("page_summaries_enabled", False)
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT, 
            f"Summary generation: {'enabled' if self.summary_enabled else 'disabled'}"
            f" | Page summaries: {'enabled' if self.page_summaries_enabled else 'disabled'}"
        )

    def _init_helper_services(self) -> None:
        """Initialize helper services for payload parsing, image config, and result building."""
        default_index = self.indexer.index or "default"
        self.payload_parser = PayloadParser(default_index=default_index)
        self.image_config_resolver = ImageConfigResolver(self.config)
        self.markdown_preprocessor = MarkdownPreprocessor(self.config)
        # ResultBuilder and MetadataEnricher are static classes, no initialization needed

    def _init_document_processor(self) -> None:
        """Compose the DocumentProcessor from already-initialised services."""
        self.doc_processor = DocumentProcessor(
            fetcher=self.fetcher,
            converter=self.converter,
            repository=self.repository,
            vision=self.vision,
            pii_manager=self.pii_manager,
            summary_generator=self.summary_generator,
            image_processor=self.image_processor,
            image_config_resolver=self.image_config_resolver,
            markdown_preprocessor=self.markdown_preprocessor,
            image_extraction_enabled=self.image_extraction_enabled,
            summary_enabled=self.summary_enabled,
            page_summaries_enabled=self.page_summaries_enabled,
        )

    def _init_indexing_orchestrator(self) -> None:
        """Compose the IndexingOrchestrator from already-initialised services."""
        self.idx_orchestrator = IndexingOrchestrator(
            chunker=self.chunker,
            embedder=self.embedder,
            indexer=self.indexer,
            image_config_resolver=self.image_config_resolver,
            config=self.config,
        )

    def _log_dependency_summary(self) -> None:
        """Log summary of resolved deployment dependencies."""
        summary = {
            "container_repository": self.dependencies.container_repository,
            "ingestion_queue": self.ingestion_queue_name,
            "dead_letter_queue": self.dead_letter_queue_name,
            "vectordb_index": self.indexer.index,
            "config_parameters": list(self.dependencies.config_parameters.keys()),
        }
        LOGGER.log_stage(ProcessLogger.STAGE_INIT, "Resolved deployment dependencies", extra_data=summary)

    def cleanup_caches(self) -> None:
        """
        Release internal caches to free memory.
        
        Call this method between document processing iterations in batch jobs
        to prevent memory buildup. Particularly useful when processing many
        documents in a loop (e.g., notebooks or Worker processes).
        
        Example usage:
            for payload in payloads:
                pipeline.process(payload)
                gc.collect()
                if idx % 5 == 0:
                    pipeline.cleanup_caches()
        """
        import gc
        
        cleared = []
        
        # Clear markdown converter caches (holds strategy-specific adapters)
        if hasattr(self, 'converter') and hasattr(self.converter, '_managers'):
            self.converter._managers.clear()
            cleared.append("markdown._managers")
        
        # Clear doc_processor's markdown converter if present
        if hasattr(self, 'doc_processor') and hasattr(self.doc_processor, 'md_converter'):
            md_conv = self.doc_processor.md_converter
            if hasattr(md_conv, '_managers'):
                md_conv._managers.clear()
                if "markdown._managers" not in cleared:
                    cleared.append("doc_processor.md_converter._managers")
        
        # Clear chunker caches (holds chunking strategy adapters)
        if hasattr(self, 'chunker') and hasattr(self.chunker, '_managers'):
            self.chunker._managers.clear()
            cleared.append("chunker._managers")
        
        # Clear embedder caches (holds embedding provider adapters)
        if hasattr(self, 'embedder') and hasattr(self.embedder, '_managers'):
            self.embedder._managers.clear()
            cleared.append("embedder._managers")
        
        # Force garbage collection to release freed memory
        gc.collect()
        
        if cleared:
            LOGGER.log_stage(
                ProcessLogger.STAGE_CLEANUP, 
                f"Cleared internal caches: {', '.join(cleared)}"
            )

    # =========================================================================
    # MAIN PROCESSING METHODS
    # =========================================================================

    def process(self, payload: Dict) -> Dict[str, Any]:
        """
        Main entry point for document processing.
        
        Supports operations: 'index' (default), 'delete'
        
        Minimal payload (web_url only):
        - {"web_url": "https://example.com/report.pdf"}
        - {"web_url": "https://example.com/page", "index_name": "my_index"}
        
        The pipeline will:
        - Auto-generate a UUID file_id if not provided
        - Use web_url as source_uri when no other source is given
        - Auto-detect file type from URL extension or content
        - Use default index if index_name is not specified
        
        index_name in payload can be:
        - A single string: "index_name"
        - A list: ["index1", "index2"]
        - A JSON string: '["index1", "index2"]'
        
        When multiple indices are specified, the document is processed once
        and indexed to all target indices.
        """
        operation = payload.get("operation", "index").lower()

        # Refresh AI model config from the cloud store when its TTL expires.
        self._maybe_refresh_config()
        
        # Normalize payload using PayloadParser (auto-generates file_id, normalizes source_uri)
        self.payload_parser.normalize_payload(payload, operation)
        
        file_id: Optional[str] = payload.get("file_id")
        target_indices = self.payload_parser.parse_target_indices(payload)
        
        start_time = time.time()

        set_file_id(file_id)

        try:
            if operation == "delete":
                if not file_id:
                    LOGGER.log_error("Delete operation requires file_id")
                    return ResultBuilder.build_error_result("Delete operation requires file_id")
                return self._handle_delete_multi(file_id, target_indices)
            
            if not file_id:
                LOGGER.log_error("Operation requires file_id")
                return ResultBuilder.build_error_result("Operation requires file_id")
            
            return self._handle_index_multi(payload, file_id, target_indices, start_time)
            
        finally:
            clear_file_id()

    def _handle_delete_multi(self, file_id: str, target_indices: List[str]) -> Dict[str, Any]:
        """
        Handle document deletion from multiple indices.
        
        Args:
            file_id: Document file ID
            target_indices: List of index names to delete from
            
        Returns:
            Result dictionary with deletion counts per index
        """
        LOGGER.log_stage(
            ProcessLogger.STAGE_DELETE, 
            f"Starting delete operation for {len(target_indices)} indices: {target_indices}"
        )
        
        total_deleted = 0
        deleted_per_index = {}
        
        # Delete graph data for this document
        if self.graph_service.enabled:
            for target_index in target_indices:
                self.graph_service.delete_graph(file_id, index_name=target_index)
        
        for target_index in target_indices:
            if self.ingestion_repo:
                self.ingestion_repo.mark_deleted(file_id, index_name=target_index)
            
            deleted = self.indexer.delete_by_file_id(file_id, index_name=target_index)
            deleted_per_index[target_index] = deleted
            total_deleted += deleted
            LOGGER.log_delete(target_index, deleted)
        
        if self.ingestion_repo:
            LOGGER.log_db_update("DELETED")
        
        return ResultBuilder.build_delete_result(
            file_id=file_id,
            target_indices=target_indices,
            deleted_per_index=deleted_per_index,
        )

    def _handle_index_multi(
        self,
        payload: Dict,
        file_id: str,
        target_indices: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Handle document indexing to multiple indices.
        
        Document processing (fetch, convert, chunk, embed) is done once,
        then the same embeddings are indexed to all target indices.
        
        Uses file-type-specific configurations for optimized processing.
        
        **Stage-resilient**: each processing stage is wrapped so that if a
        later stage fails (e.g., embedding, indexing), the partial results
        from earlier stages (markdown, chunks) are preserved and a
        comprehensive failure summary is logged at the end.
        
        Args:
            payload: Request payload
            file_id: Document file ID
            target_indices: List of index names to index to
            start_time: Processing start time
            
        Returns:
            Result dictionary with indexing details per index
        """
        LOGGER.start_processing(file_id, payload)
        clear_ingestion_events()
        LOGGER.log_stage(
            ProcessLogger.STAGE_INIT,
            f"Indexing to {len(target_indices)} indices: {target_indices}"
        )

        # Resolve processing strategies with file-type awareness
        # (done before index creation so payload vector_dimension is validated early)
        primary_index = target_indices[0]
        strategies = self.strategy_resolver.resolve_all(payload, file_id)
        
        # Get full file-type-specific configuration
        file_type_config = self.strategy_resolver.resolve_full_config(payload, file_id)
        resolved_file_type = file_type_config.get("_resolved_file_type", "default")

        # Ensure all indices exist (including the default — it may have been
        # deleted externally, so we must NOT skip it).
        # Apply payload vector_dimension override when present.
        vectordb_cfg = self.config.get("vectordb", {})
        payload_dimension = strategies.get("vector_dimension")
        if payload_dimension is not None:
            vectordb_cfg = {**vectordb_cfg, "vector_dimension": payload_dimension}
        for target_index in target_indices:
            self.indexer.ensure_index_from_config(vectordb_cfg, index_name=target_index)
        
        LOGGER.log_stage(
            ProcessLogger.STAGE_INPUT,
            f"Using file-type configuration: {resolved_file_type}",
            extra_data={"file_type": resolved_file_type}
        )
        
        self.strategy_resolver.update_db_strategies(file_id, strategies, primary_index)

        # -----------------------------------------------------------------
        # Stage-resilient processing: track completed stages and partial
        # results so that a failure in a late stage does NOT discard data
        # produced by earlier stages.
        # -----------------------------------------------------------------
        completed_stages: List[str] = []
        stage_errors: List[Dict[str, str]] = []

        # Partial results — accumulated as stages succeed
        markdown: Optional[str] = None
        markdown_location: Optional[str] = None
        image_manifest = None
        fetched_bundle = None
        chunks: List = []
        embeddings: List = []
        deleted_per_index: Dict[str, int] = {}
        indexed_per_index: Dict[str, int] = {}
        graph_result: Dict = {}
        image_chunks_count = 0
        infographic_chunks_count = 0

        try:
            # ------ Stage: Prepare ingestion-run metadata ------
            # Robust failsafe re-ingestion strategy ("index-then-sweep"):
            #
            #   1. Stamp every new chunk with a unique ``ingestion_run_id``.
            #   2. Index the new chunks FIRST (old chunks remain live and
            #      serve queries the whole time).
            #   3. After successful indexing, sweep chunks for the same
            #      ``file_id`` whose ``ingestion_run_id`` differs from the
            #      current one (including legacy chunks with no run_id).
            #
            # Benefits over the old "delete-then-index" order:
            #   * If processing/indexing fails midway, the old chunks are
            #     still queryable — no data-loss window.
            #   * The sweep is 404-tolerant, so eventual consistency
            #     stops producing noisy false-error logs.
            #   * Retries naturally clean up partial state from prior runs.
            ingestion_run_id = uuid.uuid4().hex
            LOGGER.log_stage(
                ProcessLogger.STAGE_INIT,
                f"Ingestion run id: {ingestion_run_id}",
                extra_data={"ingestion_run_id": ingestion_run_id},
            )

            metadata = extract_base_metadata(payload)
            # Keep metadata file_type aligned with canonical resolver output
            # (e.g., payload may say "document" while resolver detects "pptx").
            metadata["file_type"] = resolved_file_type
            metadata["target_indices"] = json.dumps(target_indices)
            metadata["resolved_file_type"] = resolved_file_type
            # Propagates to every text/image chunk (base_metadata is copied
            # into each chunk) so downstream citation UIs can render
            # "Slide N" for PPT/PPTX instead of assuming "Page N".
            metadata["location_label"] = resolve_location_label(resolved_file_type)
            metadata["ingestion_run_id"] = ingestion_run_id

            # ------ Stage: Document processing (fetch → convert → vision → PII → summary → store) ------
            try:
                markdown, markdown_location, image_manifest, fetched_bundle = self.doc_processor.process(
                    payload, file_id, metadata, strategies, file_type_config
                )
                completed_stages.append("document_processing")
                if markdown_location:
                    metadata["markdown_location"] = markdown_location
            except Exception as e:
                stage_errors.append({
                    "stage": "document_processing",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_error(
                    f"Document processing failed — no markdown produced",
                    exception=e,
                )
                # Cannot continue without markdown
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                )
                self._handle_processing_error(file_id, primary_index, start_time, e)
                raise

            # ------ Stage: Infographic classification (parallel) ------
            # The unified pre-scan inside the parallel hybrid adapter already
            # detects infographic pages from text/image coverage analysis.
            # When those results are available we use them directly, saving
            # a separate Vision-LLM-per-page InfographicClassifier pass
            # (which re-opens the PDF and renders every page).
            # Fall back to the full classifier only when the adapter did not
            # provide pre-scan results (e.g. non-PDF files or Docling-only).
            infographic_future = None
            infographic_executor = None
            prescan_infographic_pages = getattr(
                self.doc_processor, "detected_infographic_pages", []
            )
            is_pdf = (
                fetched_bundle is not None
                and fetched_bundle.document_path.lower().endswith(".pdf")
            )
            if prescan_infographic_pages:
                # Pre-scan already classified — no extra Vision LLM calls needed
                LOGGER.log_stage(
                    ProcessLogger.STAGE_CHUNK,
                    f"Using pre-scan infographic detection: "
                    f"{len(prescan_infographic_pages)} infographic pages "
                    f"{prescan_infographic_pages}",
                )
            elif is_pdf and self.infographic_classifier:
                infographic_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infographic")
                infographic_future = infographic_executor.submit(
                    self.infographic_classifier.classify_pdf,
                    fetched_bundle.document_path,
                )
                LOGGER.log_stage(
                    ProcessLogger.STAGE_CHUNK,
                    "Infographic classification submitted (running in parallel with chunking)",
                )

            # ------ Stage: Chunking ------
            try:
                markdown_for_chunking = self.doc_processor.prepare_for_chunking(
                    markdown, payload, file_type_config
                )

                # Extract total page count from page markers before chunking.
                # This is stored in each chunk's metadata so the frontend can
                # cap page_range_end (±2 offset) at the actual document length.
                from rag_lib.utils.page_markers import extract_all_page_numbers
                all_pages = extract_all_page_numbers(markdown_for_chunking)
                if all_pages:
                    metadata["total_pages"] = str(max(int(p) for p in all_pages))

                chunks = self.idx_orchestrator.chunk_document(
                    file_id, markdown_for_chunking, metadata,
                    strategies["chunking_strategy"], file_type_config
                )
                # Enrich chunk metadata
                for chunk in chunks:
                    if "chunk_type" not in chunk.metadata:
                        chunk.metadata["chunk_type"] = "text"
                    if "chunking_strategy" not in chunk.metadata:
                        chunk.metadata["chunking_strategy"] = strategies["chunking_strategy"]
                    # Always stamp the current ingestion_run_id so the
                    # post-index sweep can distinguish new chunks from
                    # any stale ones still in the index.
                    chunk.metadata["ingestion_run_id"] = ingestion_run_id

                # Collect infographic classification results and tag chunks
                infographic_map: Dict[int, bool] = {}
                if prescan_infographic_pages:
                    # Build map from pre-scan results (already available)
                    infographic_map = {p: True for p in prescan_infographic_pages}
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_CHUNK,
                        f"Infographic classification (pre-scan): "
                        f"{len(prescan_infographic_pages)} infographic pages "
                        f"{prescan_infographic_pages}",
                    )
                elif infographic_future is not None:
                    try:
                        infographic_map = infographic_future.result(timeout=300)
                        if infographic_executor is not None:
                            infographic_executor.shutdown(wait=False)
                        infographic_pages_detected = sorted(
                            p for p, v in infographic_map.items() if v
                        )
                        LOGGER.log_stage(
                            ProcessLogger.STAGE_CHUNK,
                            f"Infographic classification complete: "
                            f"{len(infographic_pages_detected)}/{len(infographic_map)} "
                            f"infographic pages {infographic_pages_detected or '(none)'}",
                        )
                    except Exception as classify_err:
                        LOGGER.log_warning(
                            ProcessLogger.STAGE_CHUNK,
                            f"Infographic classification failed (non-fatal, "
                            f"chunks will not be tagged): {classify_err}",
                        )
                        infographic_map = {}

                # Tag every chunk with infographic metadata based on its page number
                for chunk in chunks:
                    page_str = chunk.metadata.get("page_number")
                    if page_str and infographic_map:
                        try:
                            page_int = int(page_str)
                            chunk.metadata["infographic"] = (
                                "true" if infographic_map.get(page_int, False) else "false"
                            )
                        except (ValueError, TypeError):
                            chunk.metadata["infographic"] = "false"
                    else:
                        chunk.metadata["infographic"] = "false"

                completed_stages.append("chunking")

                # --- Early submit: infographic page chunks (runs in background) ---
                # All inputs are ready now (infographic_map, fetched_bundle, metadata).
                # Submit to a thread so the work overlaps with embedding/indexing.
                infographic_chunk_future = None
                infographic_chunk_executor = None
                vision_llm_detected = sorted(
                    p for p, v in infographic_map.items() if v
                ) if infographic_map else []
                explicit_infographic = (
                    file_type_config.get("image_extraction", {}).get("infographic_pages")
                    or payload.get("infographic_pages")
                    or []
                )
                early_infographic_pages = sorted(
                    set(vision_llm_detected) | set(explicit_infographic)
                )
                if (
                    early_infographic_pages
                    and fetched_bundle
                    and self.image_repository
                ):
                    infographic_chunk_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="infographic_chunks"
                    )
                    infographic_chunk_future = infographic_chunk_executor.submit(
                        self.idx_orchestrator.create_and_index_infographic_chunks,
                        file_id=file_id,
                        file_path=fetched_bundle.document_path,
                        infographic_pages=early_infographic_pages,
                        image_repository=self.image_repository,
                        metadata=metadata,
                        strategies=strategies,
                        target_indices=target_indices,
                        region=self.region,
                    )
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_CHUNK,
                        f"Infographic page chunk creation submitted in background "
                        f"for pages {early_infographic_pages}",
                    )
            except Exception as e:
                stage_errors.append({
                    "stage": "chunking",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_error(
                    f"Chunking failed — markdown was produced but could not be chunked",
                    exception=e,
                )
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                )
                self._handle_processing_error(file_id, primary_index, start_time, e)
                raise

            # ------ Stage: Image linking ------
            image_cfg = file_type_config.get("image_extraction", {})
            try:
                should_link = image_cfg.get(
                    "link_images_to_chunks",
                    self.image_config_resolver.should_link_images_to_chunks(payload)
                )
                if image_manifest and should_link:
                    chunks = ImageChunkLinker.link_images_to_chunks(
                        chunks=chunks,
                        image_manifest=image_manifest,
                        max_images_per_chunk=image_cfg.get(
                            "max_images_per_chunk",
                            self.image_config_resolver.get_max_images_per_chunk(payload)
                        ),
                    )

                # For non-PDF files: tag text chunks whose linked media contain
                # at least one infographic (classification set during description).
                if not is_pdf and image_manifest:
                    infographic_image_ids = set()
                    for img in image_manifest.get("images", []):
                        if isinstance(img, dict):
                            img_id = img.get("image_id")
                            is_info = img.get("is_infographic", False)
                        else:
                            img_id = getattr(img, "image_id", None)
                            is_info = getattr(img, "is_infographic", False)
                        if is_info and img_id:
                            infographic_image_ids.add(img_id)
                    if infographic_image_ids:
                        import json as _json
                        for chunk in chunks:
                            # Only upgrade — never downgrade a chunk already tagged true
                            if chunk.metadata.get("infographic") == "true":
                                continue
                            media_raw = chunk.metadata.get("media")
                            if media_raw:
                                try:
                                    media_obj = _json.loads(media_raw) if isinstance(media_raw, str) else media_raw
                                    if media_obj.get("infographics"):
                                        chunk.metadata["infographic"] = "true"
                                except Exception:
                                    pass
                        LOGGER.log_stage(
                            ProcessLogger.STAGE_CHUNK,
                            f"Web infographic tagging: {len(infographic_image_ids)} infographic "
                            f"image(s) detected — tagged linked text chunks",
                        )

                completed_stages.append("image_linking")
            except Exception as e:
                # Non-fatal: log and continue — chunks were already created
                stage_errors.append({
                    "stage": "image_linking",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_VISION,
                    f"Image linking failed (non-fatal, continuing): {e}",
                )

            # ------ Stage: Graph knowledge base ------
            try:
                if self.graph_service.enabled:
                    doc_summary = metadata.get("document_summary", "")
                    graph_result = self.graph_service.build_graph(
                        chunks=chunks,
                        file_id=file_id,
                        index_name=primary_index,
                        document_summary=doc_summary,
                    )
                completed_stages.append("graph_build")
            except Exception as e:
                # Non-fatal: graph enrichment failure should not block indexing
                stage_errors.append({
                    "stage": "graph_build",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_INDEX,
                    f"Graph knowledge base build failed (non-fatal, continuing): {e}",
                )

            # ------ Stage: Embedding ------
            try:
                embeddings = self.idx_orchestrator.embed_chunks(
                    chunks, strategies["embedding_provider"],
                    vector_dimension=strategies.get("vector_dimension"),
                )
                completed_stages.append("embedding")
            except Exception as e:
                stage_errors.append({
                    "stage": "embedding",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_error(
                    f"Embedding failed — {len(chunks)} chunks were created but could not be embedded",
                    exception=e,
                )
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                )
                self._handle_processing_error(file_id, primary_index, start_time, e)
                raise

            # ------ Stage: Image relevance scoring (cosine similarity) ------
            try:
                enable_relevance_scoring = image_cfg.get("enable_relevance_scoring", True)
                if image_manifest and enable_relevance_scoring and embeddings:
                    relevance_threshold = image_cfg.get("relevance_threshold", 0.30)
                    embed_fn = lambda desc_chunks: self.idx_orchestrator.embed_chunks(
                        desc_chunks, strategies["embedding_provider"],
                        vector_dimension=strategies.get("vector_dimension"),
                    )
                    scored = ImageRelevanceScorer.score(
                        embeddings=embeddings,
                        embed_fn=embed_fn,
                        relevance_threshold=relevance_threshold,
                    )
                    if scored:
                        LOGGER.log_stage(
                            ProcessLogger.STAGE_CHUNK,
                            f"Image relevance scored for {scored} chunks (threshold={relevance_threshold})",
                        )
                completed_stages.append("image_relevance_scoring")
            except Exception as e:
                # Non-fatal: relevance scoring failure should not block indexing
                stage_errors.append({
                    "stage": "image_relevance_scoring",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_CHUNK,
                    f"Image relevance scoring failed (non-fatal, continuing): {e}",
                )

            # ------ Stage: Index to all target indices ------
            try:
                for target_index in target_indices:
                    self.idx_orchestrator.index_embeddings(embeddings, target_index)
                    indexed_per_index[target_index] = len(embeddings)
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_INDEX,
                        f"Indexed {len(embeddings)} text chunks to {target_index}"
                    )
                completed_stages.append("indexing")
            except Exception as e:
                stage_errors.append({
                    "stage": "indexing",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_error(
                    f"Indexing failed — {len(embeddings)} embeddings produced but "
                    f"only {len(indexed_per_index)}/{len(target_indices)} indices written",
                    exception=e,
                )
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                )
                self._handle_processing_error(file_id, primary_index, start_time, e)
                raise

            # ------ Stage: Image relevance tagging ------
            try:
                enable_relevance_tagging = image_cfg.get("enable_relevance_tagging", False)
                if image_manifest and enable_relevance_tagging and chunks:
                    self.idx_orchestrator.tag_image_relevance(
                        image_manifest=image_manifest,
                        chunks=chunks,
                        region=self.region,
                    )
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_CHUNK,
                        "Image relevance tagging complete",
                    )
                completed_stages.append("relevance_tagging")
            except Exception as e:
                # Non-fatal
                stage_errors.append({
                    "stage": "relevance_tagging",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_VISION,
                    f"Image relevance tagging failed (non-fatal): {e}",
                )

            # ------ Stage: Image chunks ------
            try:
                create_image_chunks = image_cfg.get(
                    "create_image_chunks",
                    self.image_config_resolver.should_create_image_chunks(payload)
                )
                if image_manifest and create_image_chunks:
                    image_chunks_count = self.idx_orchestrator.create_and_index_image_chunks(
                        file_id, image_manifest, metadata, strategies, target_indices
                    )
                    # Update indexed counts to include image chunks
                    for target_index in target_indices:
                        indexed_per_index[target_index] = indexed_per_index.get(target_index, 0) + image_chunks_count
                completed_stages.append("image_chunks")
            except Exception as e:
                # Non-fatal: text chunks are already indexed
                stage_errors.append({
                    "stage": "image_chunks",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_INDEX,
                    f"Image chunk creation/indexing failed (non-fatal, text chunks already indexed): {e}",
                )

            # ------ Stage: Infographic page chunks (collect background result) ------
            try:
                if infographic_chunk_future is not None:
                    # Collect the result from the background task submitted after chunking
                    infographic_chunks_count = infographic_chunk_future.result(timeout=300)
                    if infographic_chunk_executor is not None:
                        infographic_chunk_executor.shutdown(wait=False)
                    LOGGER.log_stage(
                        ProcessLogger.STAGE_INDEX,
                        f"Created {infographic_chunks_count} infographic chunks for pages {early_infographic_pages}",
                    )

                # Web / HTML infographic image chunks (non-PDF path)
                elif not is_pdf and image_manifest:
                    web_infographic_count = self.idx_orchestrator.create_and_index_web_infographic_chunks(
                        file_id=file_id,
                        image_manifest=image_manifest,
                        metadata=metadata,
                        strategies=strategies,
                        target_indices=target_indices,
                    )
                    if web_infographic_count:
                        infographic_chunks_count = web_infographic_count
                        LOGGER.log_stage(
                            ProcessLogger.STAGE_INDEX,
                            f"Created {web_infographic_count} web infographic chunks",
                        )

                # Update indexed counts to include infographic chunks
                if infographic_chunks_count > 0:
                    for target_index in target_indices:
                        indexed_per_index[target_index] = indexed_per_index.get(target_index, 0) + infographic_chunks_count

                completed_stages.append("infographic_chunks")
            except Exception as e:
                # Non-fatal
                stage_errors.append({
                    "stage": "infographic_chunks",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_INDEX,
                    f"Infographic chunk creation failed (non-fatal): {e}",
                )

            # ------ Stage: Sweep stale chunks (index-then-sweep) ------
            # Now that ALL new chunks (text + image + infographic) are safely
            # indexed for every target index, remove any chunks for the same
            # file_id whose ingestion_run_id does not match the current run.
            # This includes:
            #   * Chunks from prior re-ingestions (older run_ids).
            #   * Legacy chunks with no ingestion_run_id field at all.
            #
            # This stage is deliberately placed AFTER all chunk types are
            # indexed so that if image/infographic indexing fails, the old
            # chunks remain queryable — no data-loss window.
            #
            # This stage is deliberately NON-FATAL. If the sweep cannot
            # complete (for example, after a transient vector database error) the new chunks are
            # already live and a subsequent re-ingestion will clean up any
            # leftovers. Failing the whole run here would just orphan more
            # chunks without improving correctness.
            try:
                total_swept = 0
                for target_index in target_indices:
                    if target_index not in indexed_per_index:
                        # Skip indices we never wrote new chunks to.
                        continue
                    swept = self.indexer.delete_stale_chunks_for_file(
                        file_id=file_id,
                        current_ingestion_run_id=ingestion_run_id,
                        index_name=target_index,
                    )
                    deleted_per_index[target_index] = swept
                    total_swept += swept
                    LOGGER.log_delete(target_index, swept)
                LOGGER.log_stage(
                    ProcessLogger.STAGE_DELETE,
                    f"Stale-chunk sweep complete: removed {total_swept} old chunks "
                    f"across {len(indexed_per_index)} indices "
                    f"(kept run_id={ingestion_run_id})",
                )
                completed_stages.append("sweep_stale_chunks")
            except Exception as e:
                stage_errors.append({
                    "stage": "sweep_stale_chunks",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                LOGGER.log_warning(
                    ProcessLogger.STAGE_DELETE,
                    f"Stale-chunk sweep failed (non-fatal — new chunks are live; "
                    f"future re-ingestion will clean up): {e}",
                )

            # ------ Stage: Database update ------
            self._update_db_completed(file_id, primary_index, chunks, markdown_location, start_time)
            completed_stages.append("db_update")

            # Log any non-fatal stage errors at the end for visibility
            if stage_errors:
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                    fatal=False,
                )

            return ResultBuilder.build_multi_index_result(
                file_id=file_id,
                deleted_per_index=deleted_per_index,
                chunks=chunks,
                target_indices=target_indices,
                indexed_per_index=indexed_per_index,
                strategies=strategies,
                image_manifest=image_manifest,
                image_chunks_count=image_chunks_count,
                infographic_chunks_count=infographic_chunks_count,
                markdown_location=markdown_location,
                graph_result=graph_result,
            )

        except Exception as e:
            # Catch-all: if an unexpected error slipped through the stage
            # wrappers, ensure the failure summary is still logged.
            if not stage_errors or stage_errors[-1].get("error") != str(e):
                stage_errors.append({
                    "stage": "unexpected",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                self._log_stage_failure_summary(
                    file_id, completed_stages, stage_errors,
                    markdown_location, chunks, indexed_per_index, start_time,
                )
                self._handle_processing_error(file_id, primary_index, start_time, e)
            raise
        finally:
            # Clean up the fetched document bundle (temp files) after the
            # entire pipeline — including infographic chunk creation — is done.
            if fetched_bundle:
                LOGGER.log_cleanup()
                self.doc_processor.fetcher.cleanup(fetched_bundle)

            # Flush accumulated ingestion log events to object storage
            doc_prefix = None
            if markdown_location and "://" in markdown_location:
                # The markdown location uses the active provider's URI scheme.
                _, _, without_scheme = markdown_location.partition("://")
                parts = without_scheme.split("/", 1)
                if len(parts) == 2:
                    key_path = parts[1]
                    if "/" in key_path:
                        doc_prefix = key_path.rsplit("/", 1)[0]

            if self.doc_processor.repository:
                flush_ingestion_log(
                    file_manager=self.doc_processor.repository.file_manager,
                    bucket=self.doc_processor.repository.bucket,
                    document_prefix=doc_prefix,
                    file_id=file_id,
                    base_prefix=self.doc_processor.repository.base_prefix or "",
                )
            else:
                clear_ingestion_events()

    # =========================================================================
    # DATABASE AND RESULT HANDLING
    # =========================================================================

    def _update_db_completed(
        self,
        file_id: str,
        target_index: str,
        chunks: list,
        markdown_location: str,
        start_time: float,
    ) -> None:
        """Update database with completion status."""
        if not self.ingestion_repo:
            return
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        duration_minutes = (time.time() - start_time) / 60.0
        
        self.ingestion_repo.mark_completed(
            file_id=file_id,
            index_name=target_index,
            chunk_id_list=chunk_ids,
            ingestion_duration_minutes=round(duration_minutes, 2),
            markdown_location=markdown_location
        )
        LOGGER.log_db_update("COMPLETED")
        LOGGER.log_stage(
            ProcessLogger.STAGE_COMPLETE, 
            f"Ingestion completed in {duration_minutes:.2f} minutes"
        )

    def _handle_processing_error(
        self,
        file_id: str,
        target_index: str,
        start_time: float,
        error: Exception,
    ) -> None:
        """Handle processing errors and update database."""
        LOGGER.log_error(f"Processing failed for file", exception=error)
        
        if self.ingestion_repo:
            duration_minutes = (time.time() - start_time) / 60.0
            self.ingestion_repo.mark_failed(
                file_id=file_id,
                error_message=str(error),
                index_name=target_index,
                ingestion_duration_minutes=round(duration_minutes, 2)
            )
            LOGGER.log_db_update("FAILED")
            LOGGER.log_stage(
                ProcessLogger.STAGE_ERROR, 
                f"Ingestion failed after {duration_minutes:.2f} minutes"
            )

    def _log_stage_failure_summary(
        self,
        file_id: str,
        completed_stages: List[str],
        stage_errors: List[Dict[str, str]],
        markdown_location: Optional[str],
        chunks: list,
        indexed_per_index: Dict[str, int],
        start_time: float,
        fatal: bool = True,
    ) -> None:
        """Log a comprehensive failure summary preserving partial result info.

        This is the key method that ensures no document extraction information
        is lost when a pipeline stage fails.  It logs:
        - Which stages completed successfully
        - Which stage(s) failed and why
        - What partial data was produced (markdown location, chunk count, etc.)

        Args:
            file_id: Document identifier.
            completed_stages: List of stage names that succeeded.
            stage_errors: List of dicts ``{"stage", "error", "error_type"}``.
            markdown_location: Object storage path to stored markdown (if produced).
            chunks: Chunks produced so far (may be empty).
            indexed_per_index: Dict of index→count for successfully indexed.
            start_time: Processing start timestamp.
            fatal: If True, the pipeline is aborting. If False, processing
                continued despite non-fatal stage errors.
        """
        duration_secs = time.time() - start_time
        severity = "FATAL" if fatal else "WARNING"

        summary = {
            "severity": severity,
            "file_id": file_id,
            "completed_stages": completed_stages,
            "failed_stages": [e["stage"] for e in stage_errors],
            "errors": stage_errors,
            "partial_results": {
                "markdown_location": markdown_location,
                "chunks_created": len(chunks),
                "indices_written": indexed_per_index,
            },
            "duration_seconds": round(duration_secs, 2),
        }

        log_level = logging.ERROR if fatal else logging.WARNING
        LOGGER.log_stage(
            ProcessLogger.STAGE_ERROR,
            f"PIPELINE {severity} SUMMARY — "
            f"completed: {completed_stages}, "
            f"failed: {[e['stage'] for e in stage_errors]}, "
            f"markdown: {'saved' if markdown_location else 'NOT produced'}, "
            f"chunks: {len(chunks)}, "
            f"indices_written: {indexed_per_index or 'none'}, "
            f"duration: {duration_secs:.1f}s",
            level=log_level,
            extra_data=summary,
        )
