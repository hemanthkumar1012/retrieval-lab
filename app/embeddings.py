from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import settings

# bge models were trained with this prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def embed_docs(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def embed_query(q: str) -> np.ndarray:
    return _embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True)


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    pairs = [(query, p) for p in passages]
    return _reranker().predict(pairs, batch_size=32).tolist()
