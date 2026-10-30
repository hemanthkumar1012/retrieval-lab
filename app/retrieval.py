import time

from .config import settings
from .db import conn
from .embeddings import embed_query, rerank_scores
from .fusion import Hit, rrf


def dense_search(query: str, strategy: str, k: int) -> list[Hit]:
    return dense_search_vec(embed_query(query), strategy, k)


def dense_search_vec(qv, strategy: str, k: int) -> list[Hit]:
    """Nearest chunks for one chunking strategy.

    All strategies share one HNSW index. pgvector applies the `strategy` filter
    *after* the index scan, so by default a query only sees the ~ef_search nearest
    chunks across every strategy and the filter then discards most of them: with six
    strategies a request for 40 chunks returns about 7. Iterative scans (pgvector
    0.8+) keep walking the index until enough rows survive the filter.

    set_config(..., true) is transaction-local, which also works through a
    transaction pooler, unlike a session-level SET.
    """
    with conn() as c:
        c.execute(
            "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
            " set_config('hnsw.ef_search', %s, true)",
            (str(settings.hnsw_ef_search),),
        )
        rows = c.execute(
            """SELECT id, doc_id, text, 1 - (embedding <=> %s) AS sim
               FROM chunks WHERE strategy = %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (qv, strategy, qv, k),
        ).fetchall()
    return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows]


def lexical_search(query: str, strategy: str, k: int) -> list[Hit]:
    # plainto_tsquery ANDs every term, which kills recall on natural-language
    # questions. Swap the operator to OR and let ts_rank_cd sort it out.
    with conn() as c:
        rows = c.execute(
            """WITH q AS (
                   SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS tq
               )
               SELECT id, doc_id, text, ts_rank_cd(tsv, q.tq, 32) AS rank
               FROM chunks, q
               WHERE strategy = %s AND tsv @@ q.tq
               ORDER BY rank DESC LIMIT %s""",
            (query, strategy, k),
        ).fetchall()
    return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows]


def rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
    scores = rerank_scores(query, [h.text for h in hits])
    for h, s in zip(hits, scores, strict=True):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def retrieve(
    query: str,
    strategy: str,
    mode: str = "hybrid_rerank",
    k: int = 5,
    candidates: int = 50,
    rerank_depth: int = 20,
) -> tuple[list[Hit], dict]:
    """Returns (hits, timings_ms). Timings are per stage so the benchmark can
    show where the latency goes."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    t: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        t[name] = (now - since) * 1000
        return now

    if mode == "dense":
        hits = dense_search(query, strategy, k)
        lap("dense", t0)
    elif mode == "bm25":
        hits = lexical_search(query, strategy, k)
        lap("lexical", t0)
    else:
        s = time.perf_counter()
        d = dense_search(query, strategy, candidates)
        s = lap("dense", s)
        lex = lexical_search(query, strategy, candidates)
        s = lap("lexical", s)
        fused = rrf({"dense": d, "lexical": lex}, weights={"lexical": settings.lexical_weight})
        s = lap("fusion", s)
        if mode == "hybrid":
            hits = fused[:k]
        else:
            # the reranker must see at least as many chunks as the caller wants back
            hits = rerank(query, fused[:max(rerank_depth, k)], k)
            lap("rerank", s)

    t["total"] = (time.perf_counter() - t0) * 1000
    return hits, t
