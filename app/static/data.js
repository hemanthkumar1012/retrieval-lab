// Measured results, copied from the bench workflow runs described in the README.
// BEIR SciFact test split, 300 queries, int8 ONNX models, recursive_256 chunking,
// GitHub-hosted runner (two shared cores). Run to run the numbers move by about 0.005.

export const RUN_NOTE = "BEIR SciFact test, 300 queries, recursive_256, int8 ONNX models, GitHub runner";

export const MODES = [
  { id: "dense", name: "Meaning", color: "var(--dense)", ndcg: 0.700, recall1: 0.571, recall5: 0.745, recall10: 0.816, mrr: 0.672, p50: 10 },
  { id: "bm25", name: "Keywords", color: "var(--lexical)", ndcg: 0.426, recall1: 0.273, recall5: 0.495, recall10: 0.582, mrr: 0.379, p50: 60 },
  { id: "hybrid", name: "Hybrid", color: "var(--fusion)", ndcg: 0.705, recall1: 0.576, recall5: 0.762, recall10: 0.818, mrr: 0.677, p50: 86 },
  { id: "hybrid_rerank", name: "Hybrid + rerank", color: "var(--rerank)", ndcg: 0.700, recall1: 0.573, recall5: 0.755, recall10: 0.816, mrr: 0.668, p50: 1398 },
];

export const METRICS = [
  { key: "ndcg", label: "nDCG@10" },
  { key: "recall5", label: "Recall@5" },
  { key: "recall10", label: "Recall@10" },
  { key: "mrr", label: "MRR@10" },
];

// nDCG@10 as the weight of the full-text list in rank fusion changes (vector list = 1)
export const WEIGHT_SWEEP = [
  { weight: 1, hybrid: 0.627, rerank: 0.703 },
  { weight: 0.5, hybrid: 0.656, rerank: 0.698 },
  { weight: 0.25, hybrid: 0.678, rerank: 0.697 },
  { weight: 0.1, hybrid: 0.705, rerank: 0.700 },
];
export const DENSE_NDCG = 0.700;

// Semantic cache: share of paraphrases answered from cache, and share of different
// questions that wrongly hit, per cosine-similarity threshold.
export const CACHE = [
  { threshold: 0.80, hit: 100.0, wrong: 25.7 },
  { threshold: 0.85, hit: 100.0, wrong: 20.3 },
  { threshold: 0.88, hit: 100.0, wrong: 17.7 },
  { threshold: 0.90, hit: 100.0, wrong: 15.0 },
  { threshold: 0.92, hit: 100.0, wrong: 13.0 },
  { threshold: 0.94, hit: 99.2, wrong: 10.7 },
  { threshold: 0.96, hit: 96.2, wrong: 6.0 },
  { threshold: 0.98, hit: 55.7, wrong: 2.7 },
  { threshold: 0.99, hit: 8.6, wrong: 0.0 },
  { threshold: 0.995, hit: 1.3, wrong: 0.0 },
];
export const CACHE_DEFAULT = 0.96;

export const PARITY = {
  embedderCosine: 0.982,
  rerankerCorrelation: 0.995,
  embedderMb: 34,
  rerankerMb: 23,
};

// nDCG@10 by chunking strategy, equal fusion weights (first run in the README)
export const CHUNKING = [
  { name: "recursive_256", dense: 0.695, rerank: 0.707 },
  { name: "fixed_256", dense: 0.702, rerank: 0.698 },
  { name: "sentence_256", dense: 0.687, rerank: 0.704 },
];
