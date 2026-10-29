import math


def dedupe_docs(ranked_doc_ids: list[str]) -> list[str]:
    """Chunk-level results to doc-level ranking, keeping first occurrence."""
    seen, out = set(), []
    for d in ranked_doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for d in ranked[:k] if d in relevant)
    return found / len(relevant)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]
