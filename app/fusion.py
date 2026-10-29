from dataclasses import dataclass, field


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    text: str
    score: float = 0.0
    ranks: dict = field(default_factory=dict)  # which retriever ranked it where


def rrf(lists: dict[str, list[Hit]], k: int = 60, weights: dict[str, float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion. Scores are on different scales between
    cosine and ts_rank, so fuse on rank instead of raw score."""
    weights = weights or {}
    merged: dict[int, Hit] = {}
    for name, hits in lists.items():
        w = weights.get(name, 1.0)
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.chunk_id, Hit(h.chunk_id, h.doc_id, h.text))
            m.score += w / (k + rank)
            m.ranks[name] = rank
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)
