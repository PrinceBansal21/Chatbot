"""Cloud-neutral RAG embeddings adapter."""
from __future__ import annotations

from typing import List, Optional, Tuple

from cloud_lib.factory import CloudFactory
from rag_lib.interfaces.chunking_manager import Chunk
from rag_lib.interfaces.embeddings_manager import EmbeddingsManager


class CloudEmbeddingsAdapter(EmbeddingsManager):
    """Adapt cloud_lib embeddings to rag_lib's chunk-oriented contract."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        output_dimension: Optional[int] = None,
        model_key: str = "embedding",
        config_provider: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.model_key = model_key
        self.output_dimension = output_dimension
        manager_kwargs = dict(kwargs)
        if model_id is not None:
            manager_kwargs["model_id"] = model_id
        else:
            manager_kwargs["model_key"] = model_key
        if output_dimension is not None:
            manager_kwargs["output_dimension"] = output_dimension
        self._manager = CloudFactory.get_embeddings_manager(**manager_kwargs)

    def embed(self, chunks: List[Chunk]) -> List[Tuple[Chunk, List[float]]]:
        text_positions = [
            index for index, chunk in enumerate(chunks)
            if not getattr(chunk, "image_base64", None)
        ]
        image_positions = [
            index for index, chunk in enumerate(chunks)
            if getattr(chunk, "image_base64", None)
        ]
        vectors: List[Optional[List[float]]] = [None] * len(chunks)

        if text_positions:
            texts = [
                chunks[index].embedding_content or chunks[index].content
                for index in text_positions
            ]
            for index, vector in zip(
                text_positions,
                self._manager.embed_documents(texts),
            ):
                vectors[index] = vector

        if image_positions:
            images = [chunks[index].image_base64 for index in image_positions]
            for index, vector in zip(
                image_positions,
                self._manager.embed_images(images),
            ):
                vectors[index] = vector

        return [
            (chunk, vector)
            for chunk, vector in zip(chunks, vectors)
            if vector is not None
        ]

    def embed_query(self, text: str) -> List[float]:
        return self._manager.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._manager.embed_documents(texts)

    def embed_images(self, images_base64: List[str]) -> List[List[float]]:
        return self._manager.embed_images(images_base64)
