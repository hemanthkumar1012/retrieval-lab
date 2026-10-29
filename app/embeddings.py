"""Embeddings and reranking on ONNX Runtime. No torch, so the whole API fits in a
serverless function.

The two models are int8-quantised ONNX exports of the originals, produced by
scripts/export_models.py and checked against the PyTorch versions in models/PARITY.md:

    embedder  BAAI/bge-small-en-v1.5                 CLS pooling, L2-normalised, 384 dims
    reranker  cross-encoder/ms-marco-MiniLM-L-6-v2   one relevance logit per (query, passage)

Ingestion, the benchmark and the live API all go through this module, so what is measured
is what is served.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings

# bge models were trained with this prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512


def _models_dir() -> Path:
    if settings.models_dir:
        return Path(settings.models_dir)
    return Path(__file__).resolve().parent.parent / "models"


class _OnnxModel:
    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding()  # pad to the longest sequence in the batch

    def __call__(self, inputs: list) -> np.ndarray:
        """inputs: strings, or (query, passage) tuples for the cross-encoder."""
        enc = self.tokenizer.encode_batch(inputs)
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        return self.session.run(None, feeds)[0]


@lru_cache(maxsize=1)
def _embedder() -> _OnnxModel:
    return _OnnxModel(_models_dir() / "bge-small-en-v1.5")


@lru_cache(maxsize=1)
def _reranker() -> _OnnxModel:
    return _OnnxModel(_models_dir() / "ms-marco-MiniLM-L-6-v2")


def _embed(texts: list[str], batch_size: int) -> np.ndarray:
    out = np.zeros((len(texts), settings.embed_dim), dtype=np.float32)
    # similar lengths together keeps padding (and compute) down
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    model = _embedder()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        cls = model([texts[i] for i in idx])[:, 0, :]  # CLS pooling
        out[idx] = cls / np.linalg.norm(cls, axis=1, keepdims=True)
    return out


def embed_docs(texts: list[str], batch_size: int = 32) -> np.ndarray:
    return _embed(texts, batch_size)


def embed_query(q: str) -> np.ndarray:
    return _embed([QUERY_PREFIX + q], 1)[0]


def embed_queries(qs: list[str], batch_size: int = 32) -> np.ndarray:
    """Same vectors embed_query produces, in bulk. Anything that must match what the live
    service compares (the semantic cache) has to go through the query prefix."""
    return _embed([QUERY_PREFIX + q for q in qs], batch_size)


def rerank_scores(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    if not passages:
        return []
    model = _reranker()
    scores: list[float] = []
    for start in range(0, len(passages), batch_size):
        pairs = [(query, p) for p in passages[start:start + batch_size]]
        scores.extend(model(pairs).reshape(-1).astype(float).tolist())
    return scores
