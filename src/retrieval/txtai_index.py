from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from openai import OpenAI
from txtai import Embeddings

from .catalog_schema import ProductDocument
from .config import AppConfig, EmbeddingConfig


_OPENAI_CLIENT: OpenAI | None = None


def openai_compatible_transform_env(texts: list[str]) -> np.ndarray:
    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL")
    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL is required for openai_compatible provider")
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)

    response = _OPENAI_CLIENT.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda entry: entry.index)]
    return np.array(vectors, dtype=np.float32)


def hashing_transform(texts: list[str], dimensions: int = 384) -> np.ndarray:
    matrix = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            index = hash(token) % dimensions
            matrix[row, index] += 1.0
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix


def _allow_external_transform() -> None:
    # txtai >=9.12 refuses string transform paths unless this is set.
    os.environ.setdefault("ALLOW_RESOLVE_TRANSFORM", "True")


def create_embeddings(config: EmbeddingConfig) -> Embeddings:
    provider = config.provider.lower().strip()
    if provider == "openai_compatible":
        _allow_external_transform()
        if config.base_url:
            os.environ["EMBEDDING_BASE_URL"] = config.base_url
        if config.api_key:
            os.environ["EMBEDDING_API_KEY"] = config.api_key
        os.environ["EMBEDDING_MODEL"] = config.model
        return Embeddings(
            {
                "method": "external",
                "transform": "retrieval.txtai_index.openai_compatible_transform_env",
                "content": True,
                "objects": True,
                "backend": "faiss",
                "batch": config.batch_size,
            }
        )

    if provider == "huggingface":
        return Embeddings(
            {
                "path": config.model,
                "content": True,
                "objects": True,
                "backend": "faiss",
                "batch": config.batch_size,
            }
        )

    if provider == "hashing":
        _allow_external_transform()
        return Embeddings(
            {
                "method": "external",
                "transform": "retrieval.txtai_index.hashing_transform",
                "content": True,
                "objects": True,
                "backend": "faiss",
                "batch": config.batch_size,
            }
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{config.provider}'. "
        "Use 'openai_compatible', 'huggingface' or 'hashing'."
    )


def build_txtai_index(products: list[ProductDocument], app_config: AppConfig) -> dict[str, Any]:
    app_config.paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    app_config.paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    app_config.paths.settings_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings(app_config.embedding)
    records = [(product.sku, product.retrieval_text(), None) for product in products]
    embeddings.index(records)
    embeddings.save(str(app_config.paths.index_path))

    metadata = {product.sku: product.search_payload() for product in products}
    with app_config.paths.metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    settings = {
        "embedding_provider": app_config.embedding.provider,
        "embedding_model": app_config.embedding.model,
        "embedding_base_url": app_config.embedding.base_url,
        "txtai_backend": "faiss",
        "txtai_content": True,
        "txtai_objects": True,
        "embedding_batch_size": app_config.embedding.batch_size,
        "feed_size": len(products),
        "index_path": str(app_config.paths.index_path),
        "metadata_path": str(app_config.paths.metadata_path),
    }
    with app_config.paths.settings_path.open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)

    return settings


def load_txtai_index(app_config: AppConfig) -> Embeddings:
    embeddings = create_embeddings(app_config.embedding)
    embeddings.load(str(app_config.paths.index_path))
    return embeddings
