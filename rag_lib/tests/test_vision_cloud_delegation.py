from unittest.mock import MagicMock

import pytest

from rag_lib.adapters.embeddings.cloud_embeddings_adapter import CloudEmbeddingsAdapter
from rag_lib.adapters.markdown.hybrid.markdown_adapter import HybridAdapter
from rag_lib.adapters.markdown.image_aware.markdown_adapter import ImageAwareMarkdownAdapter
from rag_lib.adapters.markdown.image_aware.models import ExtractedImage
from rag_lib.adapters.markdown.parallel_hybrid_pdf.markdown_adapter import ParallelHybridPdfAdapter
from rag_lib.adapters.markdown.pptx_vision.markdown_adapter import PptxVisionAdapter
from rag_lib.adapters.markdown.vision_llm.markdown_adapter import VisionLLMAdapter
from rag_lib.interfaces.chunking_manager import Chunk


def _configured_adapter(adapter_type, manager):
    adapter = object.__new__(adapter_type)
    adapter.model_id = "vision-model"
    adapter.max_tokens = 900
    adapter.temperature = 0.15
    adapter.region = "test-region"
    adapter._get_llm_adapter = MagicMock(return_value=manager)
    return adapter


def test_pptx_vision_delegates_to_cloud_vision_operation():
    manager = MagicMock()
    manager.converse_vision.return_value = "pptx result"
    adapter = _configured_adapter(PptxVisionAdapter, manager)

    result = adapter._call_vision_llm(b"\xff\xd8\xffjpeg", "describe")

    assert result == "pptx result"
    manager.converse_vision.assert_called_once_with(
        model_id="vision-model",
        prompt="describe",
        image_bytes=b"\xff\xd8\xffjpeg",
        image_format="jpeg",
        max_tokens=900,
        temperature=0.15,
    )


def test_document_vision_delegates_to_cloud_vision_operation():
    manager = MagicMock()
    manager.converse_vision.return_value = "document result"
    adapter = _configured_adapter(VisionLLMAdapter, manager)

    result = adapter._call_vision_model(b"png-bytes", "extract")

    assert result == "document result"
    manager.converse_vision.assert_called_once_with(
        model_id="vision-model",
        prompt="extract",
        image_bytes=b"png-bytes",
        image_format="png",
        max_tokens=900,
        temperature=0.15,
    )


@pytest.mark.parametrize("adapter_type", [HybridAdapter, ParallelHybridPdfAdapter])
def test_hybrid_vision_model_is_explicit_model_id(adapter_type):
    adapter = adapter_type(vision_model="vision-deployment")

    assert adapter.vision_config["model_id"] == "vision-deployment"
    assert "model" not in adapter.vision_config


@pytest.mark.parametrize("adapter_type", [VisionLLMAdapter, PptxVisionAdapter])
def test_explicit_vision_alias_resolves_from_config(monkeypatch, adapter_type):
    monkeypatch.setattr(
        adapter_type,
        "_get_supported_models",
        classmethod(lambda cls: {"configured-alias": {"model_id": "provider-model", "max_tokens": 321}}),
    )

    adapter = adapter_type(model="configured-alias")

    assert adapter.model_id == "provider-model"
    assert adapter.max_tokens == 321


def test_image_aware_description_delegates_to_cloud_vision_operation():
    manager = MagicMock()
    manager.converse_vision.return_value = "image result"
    adapter = object.__new__(ImageAwareMarkdownAdapter)
    adapter._llm_adapter = manager
    adapter.description_model = "vision-model"
    adapter.region = "test-region"

    image = ExtractedImage(
        image_id="image-1",
        page_number=1,
        image_bytes=b"jpeg-bytes",
        format="jpg",
    )
    result = adapter._generate_image_description(image)

    assert result == "image result"
    call = manager.converse_vision.call_args.kwargs
    assert call["model_id"] == "vision-model"
    assert call["image_bytes"] == b"jpeg-bytes"
    assert call["image_format"] == "jpeg"
    assert call["max_tokens"] == 1500
    assert call["temperature"] == 0
    assert "document image analysis expert" in call["prompt"]


def test_image_storage_delegates_uri_construction_to_cloud_file_manager():
    manager = MagicMock()
    manager.build_uri.return_value = "provider://images/doc/image-1.png"
    manager.get_object_url.return_value = "https://objects.example/image-1.png"
    adapter = object.__new__(ImageAwareMarkdownAdapter)
    adapter.storage_bucket = "images"
    adapter.path_prefix = "documents"
    adapter._file_manager = manager
    image = ExtractedImage(
        image_id="image-1",
        page_number=1,
        image_bytes=b"png-bytes",
    )

    assert adapter._store_images_cloud([image], "doc") == [image]
    manager.put_object.assert_called_once_with(
        "images",
        "documents/doc/image-1.png",
        b"png-bytes",
        content_type="image/png",
    )
    manager.build_uri.assert_called_once_with("images", "documents/doc/image-1.png")
    assert image.storage_path == "provider://images/doc/image-1.png"
    assert image.storage_url == "https://objects.example/image-1.png"


def test_cloud_embedding_adapter_uses_cloud_embeddings_manager(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "provider-a")
    manager = MagicMock()
    manager.embed_documents.return_value = [[0.1], [0.2]]
    manager.embed_query.return_value = [0.3]
    get_manager = MagicMock(return_value=manager)
    monkeypatch.setattr(
        "rag_lib.adapters.embeddings.cloud_embeddings_adapter."
        "CloudFactory.get_embeddings_manager",
        get_manager,
    )

    adapter = CloudEmbeddingsAdapter(
        model_id="provider-embedding-model",
        output_dimension=512,
    )
    chunks = [
        Chunk(content="first", chunk_id="1", file_id="file", order=0, metadata={}),
        Chunk(content="second", chunk_id="2", file_id="file", order=1, metadata={}),
    ]

    assert adapter.embed(chunks) == [(chunks[0], [0.1]), (chunks[1], [0.2])]
    assert adapter.embed_query("query") == [0.3]
    get_manager.assert_called_once_with(
        model_id="provider-embedding-model",
        output_dimension=512,
    )


@pytest.mark.parametrize("adapter_type", [VisionLLMAdapter, PptxVisionAdapter])
def test_vision_binding_is_delegated_to_cloud_lib(
    monkeypatch,
    adapter_type,
):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_VISION_DEPLOYMENT", "vision-deployment")
    monkeypatch.setattr(
        adapter_type,
        "_get_supported_models",
        classmethod(
            lambda cls: {
                "vision-large": {"model_id": "provider-large", "max_tokens": 4096},
                    "claude-haiku": {"model_id": None, "max_tokens": 4096},
            }
        ),
    )
    monkeypatch.setattr(
        "rag_lib.model_config.get_model_spec",
        MagicMock(return_value={"model_id": None, "region": None}),
    )
    manager = MagicMock()
    get_manager = MagicMock(return_value=manager)
    monkeypatch.setattr(
        "cloud_lib.factory.CloudFactory.get_llm_manager",
        get_manager,
    )

    adapter = adapter_type()

    assert adapter.model_id is None
    assert adapter.region is None
    assert adapter._get_llm_adapter() is manager
    get_manager.assert_called_once_with(model_key="vision")
