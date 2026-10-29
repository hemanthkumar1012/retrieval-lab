"""Export the two models to int8 ONNX and measure what quantisation costs.

    python scripts/export_models.py            # writes models/ and models/PARITY.md

Needs torch, optimum[exporters], sentence-transformers, onnxruntime and datasets; none of
them are runtime dependencies. Run by .github/workflows/export-models.yml, which commits
the result. The parity check imports app.embeddings, so it exercises the exact code the API
runs (tokenisation, pooling, normalisation), not a re-implementation.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SOURCES = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "feature-extraction"),
    "ms-marco-MiniLM-L-6-v2": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "text-classification"),
}


def export(name: str, repo: str, task: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = MODELS / name
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["optimum-cli", "export", "onnx", "--model", repo, "--task", task, "--opset", "17", tmp],
            check=True,
        )
        out.mkdir(parents=True)
        shutil.copy(Path(tmp) / "tokenizer.json", out / "tokenizer.json")
        quantize_dynamic(Path(tmp) / "model.onnx", out / "model.onnx", weight_type=QuantType.QInt8)
    print(f"{name}: {(out / 'model.onnx').stat().st_size / 1e6:.1f} MB int8")


def sample_passages(n: int) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    docs = [f"{d['title']}. {d['text']}" for d in corpus.select(range(n))]
    qs = [q["text"] for q in queries.select(range(n))]
    return qs, docs


def parity(n: int = 200) -> str:
    os.environ["MODELS_DIR"] = str(MODELS)
    sys.path.insert(0, str(ROOT))
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from app import embeddings as onnx

    qs, docs = sample_passages(n)
    st = SentenceTransformer(SOURCES["bge-small-en-v1.5"][0])
    ref_d = st.encode(docs, normalize_embeddings=True, batch_size=32)
    ref_q = st.encode([onnx.QUERY_PREFIX + q for q in qs], normalize_embeddings=True, batch_size=32)
    cos_d = (ref_d * onnx.embed_docs(docs)).sum(axis=1)
    cos_q = (ref_q * onnx.embed_queries(qs)).sum(axis=1)

    # does the int8 model rank documents for a query the way fp32 does?
    sims_ref = ref_q @ ref_d.T
    sims_q8 = onnx.embed_queries(qs) @ onnx.embed_docs(docs).T
    top1 = float((sims_ref.argmax(axis=1) == sims_q8.argmax(axis=1)).mean())

    ce = CrossEncoder(SOURCES["ms-marco-MiniLM-L-6-v2"][0])
    pairs = [(qs[i % len(qs)], docs[(i * 7) % len(docs)]) for i in range(n)]
    # raw logits (Identity), the same quantity the ONNX reranker returns
    ref_s = np.array(ce.predict(pairs, batch_size=16, activation_fct=torch.nn.Identity()))
    q8_s = np.array([onnx.rerank_scores(q, [p])[0] for q, p in pairs])
    corr = float(np.corrcoef(ref_s.reshape(-1), q8_s)[0, 1])

    lines = [
        "# int8 ONNX vs original PyTorch weights",
        "",
        f"Measured on {n} SciFact passages and queries by `scripts/export_models.py`.",
        "",
        "| check | result |",
        "|---|---|",
        f"| embedder, document cosine to fp32 (mean / min) | {cos_d.mean():.4f} / {cos_d.min():.4f} |",
        f"| embedder, query cosine to fp32 (mean / min) | {cos_q.mean():.4f} / {cos_q.min():.4f} |",
        f"| embedder, nearest passage per query agrees with fp32 | {top1:.1%} |",
        f"| reranker, score correlation with fp32 (Pearson) | {corr:.4f} |",
        "",
    ]
    report = "\n".join(lines)
    (MODELS / "PARITY.md").write_text(report)
    print(report)
    assert cos_d.mean() > 0.97 and cos_q.mean() > 0.97, "embedder drifted too far from fp32"
    assert corr > 0.95, "reranker drifted too far from fp32"
    return report


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    for name, (repo, task) in SOURCES.items():
        export(name, repo, task)
    parity()
