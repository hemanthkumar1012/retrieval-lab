from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .config import settings

if TYPE_CHECKING:  # heavy imports stay lazy so SQL-only tests don't need torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

# bge models were trained with this prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def _reranker() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def embed_docs(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def embed_query(q: str) -> np.ndarray:
    return _embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True)


def embed_queries(qs: list[str], batch_size: int = 64) -> np.ndarray:
    """Same vectors embed_query produces, in bulk. Anything that must match what the live
    service compares (the semantic cache) has to go through the query prefix."""
    return _embedder().encode(
        [QUERY_PREFIX + q for q in qs],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    pairs = [(query, p) for p in passages]
    return _reranker().predict(pairs, batch_size=32).tolist()
