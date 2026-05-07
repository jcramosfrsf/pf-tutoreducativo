from __future__ import annotations

import numpy as np

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_default_embedding_fn: DefaultEmbeddingFunction | None = None


def _get_default_embedding_fn() -> DefaultEmbeddingFunction:
    global _default_embedding_fn
    if _default_embedding_fn is None:
        _default_embedding_fn = DefaultEmbeddingFunction()
    return _default_embedding_fn


def get_embedding(text: str) -> list[float]:
    """Genera el embedding de un texto usando DefaultEmbeddingFunction."""
    fn = _get_default_embedding_fn()
    return fn([text])[0].tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos."""
    fn = _get_default_embedding_fn()
    return [e.tolist() for e in fn(texts)]


def get_embedding_function() -> DefaultEmbeddingFunction:
    """Retorna la función de embedding para usar con ChromaDB."""
    return _get_default_embedding_fn()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))