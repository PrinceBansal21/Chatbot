import importlib
import os
from typing import Optional, Any, Dict, Tuple
from rag_lib.interfaces.markdown_manager import MarkdownManager
from rag_lib.interfaces.chunking_manager import ChunkingManager
from rag_lib.interfaces.embeddings_manager import EmbeddingsManager
from rag_lib.interfaces.pii_manager import PIIManager
from rag_lib.interfaces.llm_manager import LLMManager
from rag_lib.interfaces.retriever_manager import RetrieverManager
from rag_lib.exceptions import ConfigurationException


# Supported cloud providers for embeddings
SUPPORTED_EMBEDDING_PROVIDERS = ("cloud",)


class RAGFactory:
    """
    Factory class to create RAG service managers based on provider configuration.
    Uses dynamic module loading to avoid hardcoded imports.
    
    Cloud-backed managers use a single logical ``cloud`` adapter. Provider
    selection and configuration are delegated to ``cloud_lib``.
    """

    # Registry mapping: category -> key -> (module_path, class_name)
    _REGISTRY: Dict[str, Dict[str, Tuple[str, str]]] = {
        "markdown": {
            "docling": ("rag_lib.adapters.markdown.docling.markdown_adapter", "DoclingAdapter"),
            "vision_llm": ("rag_lib.adapters.markdown.vision_llm.markdown_adapter", "VisionLLMAdapter"),
            "hybrid": ("rag_lib.adapters.markdown.hybrid.markdown_adapter", "HybridAdapter"),
            "image_aware": ("rag_lib.adapters.markdown.image_aware.markdown_adapter", "ImageAwareMarkdownAdapter"),
            "pptx_vision": ("rag_lib.adapters.markdown.pptx_vision.markdown_adapter", "PptxVisionAdapter"),
            "parallel_hybrid_pdf": ("rag_lib.adapters.markdown.parallel_hybrid_pdf.markdown_adapter", "ParallelHybridPdfAdapter"),
        },
        "chunking": {
            "semantic": ("rag_lib.adapters.chunking.semantic_chunking_adapter", "SemanticChunkingAdapter"),
            "recursive": ("rag_lib.adapters.chunking.recursive_chunking_adapter", "RecursiveChunkingAdapter"),
            "hierarchical": ("rag_lib.adapters.chunking.hierarchical_chunking_adapter", "HierarchicalChunkingAdapter"),
            "contextual": ("rag_lib.adapters.chunking.contextual_retrieval_adapter", "ContextualRetrievalAdapter"),
            "spreadsheet": ("rag_lib.adapters.chunking.spreadsheet_chunking_adapter", "SpreadsheetChunkingAdapter"),
        },
        "embeddings": {
            "cloud": ("rag_lib.adapters.embeddings.cloud_embeddings_adapter", "CloudEmbeddingsAdapter"),
        },
        "pii": {
            "presidio": ("rag_lib.adapters.pii.presidio_adapter", "PresidioAdapter"),
        },
        "llm": {
            "cloud": ("rag_lib.adapters.llm.cloud_llm_adapter", "CloudLLMAdapter"),
        },
        "retriever": {
            "standard": ("rag_lib.adapters.retriever.standard_retriever", "StandardRetriever"),
            "hierarchical": ("rag_lib.adapters.retriever.hierarchical_retriever", "HierarchicalRetriever"),
            "hybrid": ("rag_lib.adapters.retriever.hybrid_retriever", "HybridRetriever"),
            "graph": ("rag_lib.adapters.retriever.graph_retriever", "GraphAugmentedRetriever"),
        }
    }

    @classmethod
    def get_cloud_provider(cls) -> str:
        """
        Detect and return the cloud provider from environment variable.
        
        Reads CLOUD_PROVIDER environment variable (case-insensitive).
        No default is applied — misconfiguration must fail loudly rather
        than silently assuming a particular provider.
        
        Returns:
            str: The configured cloud provider name.

        Raises:
            ConfigurationException: If CLOUD_PROVIDER is not set.
        """
        provider = os.environ.get("CLOUD_PROVIDER", "").strip().lower()
        if not provider:
            raise ConfigurationException(
                "CLOUD_PROVIDER environment variable is not set. "
                "Set it to the provider configured by cloud_lib."
            )
        return provider

    @classmethod
    def _default_provider_for_cloud(cls, cloud_provider: str) -> str:
        """Return the logical adapter key after verifying provider configuration."""
        if not cloud_provider:
            raise ConfigurationException("A cloud provider must be configured.")
        return "cloud"

    @classmethod
    def get_default_embedding_provider(cls) -> str:
        """
        Get the default embedding provider based on the cloud provider.
        
        Returns:
            str: The embedding adapter key ('cloud')

        Raises:
            ConfigurationException: If CLOUD_PROVIDER is unset or unmapped.
        """
        return cls._default_provider_for_cloud(cls.get_cloud_provider())

    @classmethod
    def get_default_llm_provider(cls) -> str:
        """
        Get the default LLM provider based on the cloud provider.

        Returns:
            str: The LLM adapter key ('cloud')

        Raises:
            ConfigurationException: If CLOUD_PROVIDER is unset or unmapped.
        """
        return cls._default_provider_for_cloud(cls.get_cloud_provider())

    @classmethod
    def _load_adapter(cls, category: str, key: str, **kwargs) -> Any:
        """Dynamically loads and instantiates an adapter."""
        key = key.lower()
        if category not in cls._REGISTRY:
            raise ConfigurationException(f"Unknown category: {category}")
        
        if key not in cls._REGISTRY[category]:
            available = list(cls._REGISTRY[category].keys())
            raise ConfigurationException(
                f"Unsupported {category} provider/strategy: '{key}'. "
                f"Available options: {', '.join(available)}"
            )
            
        module_path, class_name = cls._REGISTRY[category][key]
        
        try:
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(**kwargs)
        except ImportError as e:
            raise ConfigurationException(
                f"Failed to import module {module_path}: {e}. "
                f"Make sure the required dependencies are installed for provider '{key}'."
            )
        except AttributeError as e:
            raise ConfigurationException(f"Class {class_name} not found in module {module_path}: {e}")
        except Exception as e:
            raise ConfigurationException(f"Failed to instantiate {class_name}: {e}")

    @staticmethod
    def get_markdown_manager(provider: str = 'docling', **kwargs) -> MarkdownManager:
        """
        Get a markdown converter manager instance.
        
        Args:
            provider: Markdown converter strategy (e.g., 'docling', 'hybrid')
            **kwargs: Provider-specific configuration
            
        Returns:
            MarkdownManager: Markdown converter instance
        """
        return RAGFactory._load_adapter("markdown", provider, **kwargs)

    @staticmethod
    def get_chunking_manager(strategy: str = 'semantic', **kwargs) -> ChunkingManager:
        """
        Get a text chunking manager instance.
        
        Args:
            strategy: Chunking strategy (e.g., 'semantic', 'recursive', 'fixed-size')
            **kwargs: Strategy-specific configuration
            
        Returns:
            ChunkingManager: Chunking manager instance
        """
        return RAGFactory._load_adapter("chunking", strategy, **kwargs)

    @staticmethod
    def get_embeddings_manager(provider: Optional[str] = None, **kwargs) -> EmbeddingsManager:
        """
        Get an embeddings manager instance.
        
        Args:
            provider: Embedding adapter key. Only ``cloud`` is supported.
                     If None, auto-detected from CLOUD_PROVIDER env var.
            **kwargs: Provider-specific configuration
            
        Returns:
            EmbeddingsManager: Embeddings manager instance
        """
        if provider is None:
            provider = RAGFactory.get_default_embedding_provider()
        return RAGFactory._load_adapter("embeddings", provider, **kwargs)

    @classmethod
    def get_pii_manager(cls, provider: str = "presidio", **kwargs) -> PIIManager:
        """
        Factory method to get a PII manager instance.
        
        Args:
            provider (str): The PII provider name (e.g., 'presidio').
            **kwargs: Configuration arguments for the provider.
            
        Returns:
            PIIManager: An instance of the requested PII manager.
        """
        return cls._load_adapter("pii", provider, **kwargs)

    @classmethod
    def get_llm_manager(cls, provider: Optional[str] = None, **kwargs) -> LLMManager:
        """
        Factory method to get an LLM manager instance for text generation.
        
        Args:
            provider: LLM adapter key. Only ``cloud`` is supported.
                     If None, defaults based on CLOUD_PROVIDER env var.
            **kwargs: Provider-specific configuration (model_id, region, etc.)
            
        Returns:
            LLMManager: An LLM manager instance for text generation.
        """
        if provider is None:
            provider = cls.get_default_llm_provider()
        return cls._load_adapter("llm", provider, **kwargs)

    @classmethod
    def get_retriever_manager(
        cls,
        strategy: str = "standard",
        vectordb_client: Any = None,
        index_name: str = None,
        embed_query_fn: Any = None,
        **kwargs
    ) -> RetrieverManager:
        """
        Factory method to get a retriever manager instance.
        
        Args:
            strategy: Retrieval strategy ('standard', 'hierarchical', 'hybrid', 'graph').
                     - 'standard': Basic vector similarity (for recursive, contextual, semantic chunking)
                     - 'hierarchical': Parent-child expansion (for hierarchical chunking)
                     - 'hybrid': Vector + keyword search
                     - 'graph': Graph-augmented retrieval
            vectordb_client: VectorDB manager instance (a cloud_lib VectorDBManager)
            index_name: Name of the index to search
            embed_query_fn: Function to convert query text to embedding vector
            **kwargs: Strategy-specific configuration
            
        Returns:
            RetrieverManager: A retriever manager instance.
            
        Example:
            # Standard retrieval
            retriever = RAGFactory.get_retriever_manager(
                strategy="standard",
                vectordb_client=vectordb,
                index_name="my-index",
                embed_query_fn=embeddings_manager.embed_query
            )
            
            # Hierarchical retrieval with parent expansion
            retriever = RAGFactory.get_retriever_manager(
                strategy="hierarchical",
                vectordb_client=vectordb,
                index_name="my-index",
                embed_query_fn=embeddings_manager.embed_query,
                expand_to_parent=True,
                include_summaries=True
            )
        """
        return cls._load_adapter(
            "retriever",
            strategy,
            vectordb_client=vectordb_client,
            index_name=index_name,
            embed_query_fn=embed_query_fn,
            **kwargs
        )
