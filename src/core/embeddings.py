"""
Shared embedding model for semantic search across the application.

Uses sentence-transformers with a configurable model (default: all-mpnet-base-v2).
The model is lazy-loaded as a singleton to avoid repeated initialization.

:mod:`embeddings` provides embedding and similarity utilities for semantic search.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.logging import get_logger

logger = get_logger("embeddings")

_model = None


def _get_model():
    """
    Lazy-load the sentence-transformers model (singleton).

    :return: Loaded SentenceTransformer model.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from src.config import get_settings

        model_name = get_settings().embedding_model
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")
    return _model


def embed(texts: list[str]) -> NDArray[np.float32]:
    """
    Encode a list of texts into embedding vectors.

    :param texts: List of input strings.
    :return: NumPy array of shape (len(texts), dim), normalized.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 768)
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(
    query_embedding: NDArray[np.float32],
    corpus_embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between a query and corpus embeddings.

    Since embeddings are normalized, this is just a dot product.

    :param query_embedding: Embedding vector for the query (1D).
    :param corpus_embeddings: Matrix of corpus embeddings (2D).
    :return: Similarity scores as a 1D NumPy array.
    """
    if corpus_embeddings.size == 0:
        return np.array([], dtype=np.float32)
    query = query_embedding.reshape(1, -1)
    return (query @ corpus_embeddings.T).flatten()
