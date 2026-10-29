# int8 ONNX vs original PyTorch weights

Measured on 200 SciFact passages and queries by `scripts/export_models.py`.

| check | result |
|---|---|
| embedder, document cosine to fp32 (mean / min) | 0.9823 / 0.9701 |
| embedder, query cosine to fp32 (mean / min) | 0.9781 / 0.9647 |
| embedder, nearest passage per query agrees with fp32 | 74.5% |
| reranker, score correlation with fp32 (Pearson) | 0.9954 |
