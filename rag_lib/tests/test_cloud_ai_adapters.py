from unittest.mock import MagicMock

import pytest

from rag_lib.adapters.embeddings.cloud_embeddings_adapter import CloudEmbeddingsAdapter
from rag_lib.exceptions import ConfigurationException
from rag_lib.factory import RAGFactory
from rag_lib.interfaces.chunking_manager import Chunk
from rag_lib import model_config


def test_cloud_embeddings_preserve_chunk_order(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    manager = MagicMock()
    manager.embed_documents.return_value = [[1.0], [3.0]]
    manager.embed_images.return_value = [[2.0]]
    manager.embed_query.return_value = [4.0]
    monkeypatch.setattr(
        "rag_lib.adapters.embeddings.cloud_embeddings_adapter."
        "CloudFactory.get_embeddings_manager",
        MagicMock(return_value=manager),
    )
    chunks = [
        Chunk(content="first", chunk_id="1", file_id="file", order=0, metadata={}),
        Chunk(
            content="image",
            chunk_id="2",
            file_id="file",
            order=1,
            metadata={},
            image_base64="base64-image",
        ),
        Chunk(
            content="third",
            embedding_content="third with context",
            chunk_id="3",
            file_id="file",
            order=2,
            metadata={},
        ),
    ]

    adapter = CloudEmbeddingsAdapter(model_id="embedding-deployment")

    assert adapter.embed(chunks) == [
        (chunks[0], [1.0]),
        (chunks[1], [2.0]),
        (chunks[2], [3.0]),
    ]
    assert adapter.embed_query("query") == [4.0]
    manager.embed_documents.assert_called_once_with(["first", "third with context"])
    manager.embed_images.assert_called_once_with(["base64-image"])


def test_factory_defaults_ai_managers_to_cloud(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    embedding_manager = MagicMock()
    llm_manager = MagicMock()
    monkeypatch.setattr(
        "rag_lib.adapters.embeddings.cloud_embeddings_adapter."
        "CloudFactory.get_embeddings_manager",
        MagicMock(return_value=embedding_manager),
    )
    monkeypatch.setattr(
        "cloud_lib.factory.CloudFactory.get_llm_manager",
        MagicMock(return_value=llm_manager),
    )

    embeddings = RAGFactory.get_embeddings_manager(model_id="embedding-deployment")
    llm = RAGFactory.get_llm_manager(model_id="chat-deployment")

    assert type(embeddings).__name__ == "CloudEmbeddingsAdapter"
    assert type(llm).__name__ == "CloudLLMAdapter"


def test_cloud_llm_normalizes_converse_response(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    manager = MagicMock()
    manager.converse.return_value = {
        "output": {"message": {"content": [{"text": "hello"}, {"text": "world"}]}},
        "usage": {"inputTokens": 3, "outputTokens": 2},
    }
    monkeypatch.setattr(
        "cloud_lib.factory.CloudFactory.get_llm_manager",
        MagicMock(return_value=manager),
    )

    adapter = RAGFactory.get_llm_manager(model_id="chat-deployment")
    result = adapter.generate("prompt", system_prompt="system")

    assert result.text == "hello world"
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert result.metadata == {"provider": "cloud", "model_key": "llm"}
    manager.converse.assert_called_once_with(
        model_id="chat-deployment",
        messages=[{"role": "user", "content": [{"text": "prompt"}]}],
        system=[{"text": "system"}],
        inference_config={"maxTokens": 1024, "temperature": 0.3},
    )


@pytest.mark.parametrize("category, provider", [
    ("embeddings", "native"),
    ("embeddings", "direct"),
    ("embeddings", "cohere"),
    ("embeddings", "azure"),
    ("embeddings", "openai"),
    ("llm", "native"),
    ("llm", "direct"),
    ("llm", "nova"),
    ("llm", "azure"),
    ("llm", "openai"),
])
def test_provider_named_ai_adapters_are_rejected(monkeypatch, category, provider):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    factory = (
        RAGFactory.get_embeddings_manager
        if category == "embeddings"
        else RAGFactory.get_llm_manager
    )

    with pytest.raises(ConfigurationException, match=f"Unsupported {category}"):
        factory(provider, model_id="model")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "provider-a")

    with pytest.raises(ConfigurationException, match="Unsupported embeddings"):
        RAGFactory.get_embeddings_manager("gcp", model_id="model")


def test_ai_model_bindings_are_delegated_to_cloud_lib(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding-deployment")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-deployment")
    embedding_factory = MagicMock(return_value=MagicMock())
    llm_manager = MagicMock()
    llm_manager.converse.return_value = {"output": {"message": {"content": []}}}
    llm_factory = MagicMock(return_value=llm_manager)
    monkeypatch.setattr(
        "rag_lib.adapters.embeddings.cloud_embeddings_adapter."
        "CloudFactory.get_embeddings_manager",
        embedding_factory,
    )
    monkeypatch.setattr(
        "cloud_lib.factory.CloudFactory.get_llm_manager",
        llm_factory,
    )

    embeddings = RAGFactory.get_embeddings_manager()
    llm = RAGFactory.get_llm_manager()
    llm.generate("prompt")

    assert embeddings.model_id is None
    assert llm.model_id is None
    embedding_factory.assert_called_once_with(
        model_key="embedding",
    )
    llm_factory.assert_called_once_with(model_key="llm")


def test_model_lookup_ignores_provider_specific_region_fallbacks(monkeypatch):
    monkeypatch.setenv("PROVIDER_REGION", "region-1")
    monkeypatch.delenv("CLOUD_REGION", raising=False)
    model_config.seed_config({})

    try:
        assert model_config.get_model_default("llm", "azure") is None
        assert model_config.get_model_spec("llm", "azure") == {
            "model_id": None,
            "region": None,
        }
        assert model_config.get_model_spec("llm", "provider-a") == {
            "model_id": None,
            "region": None,
        }
    finally:
        model_config._CONFIG_CACHE = None


def test_model_refresh_uses_canonical_ai_models_config(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    monkeypatch.setenv("CLOUD_REGION", "eastus")
    fetch_config = MagicMock(return_value={"model_defaults": {"llm": {}}})
    monkeypatch.setattr(
        "cloud_lib.fetch_parameter_service_config",
        fetch_config,
    )

    try:
        data = model_config.refresh_config_from_cloud_config_store()
    finally:
        model_config._CONFIG_CACHE = None

    assert data == {"model_defaults": {"llm": {}}}
    fetch_config.assert_called_once_with("ai-models", region="eastus")