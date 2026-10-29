"""Runs the retrieval SQL against a real Postgres with pgvector.

    TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest tests/test_sql_integration.py

Skipped when TEST_DATABASE_URL is not set. Everything happens in a throwaway schema
(`_t_rl`) that is dropped afterwards; the `public` tables are never touched.

That schema deliberately has only the HNSW index on `chunks` (the btree on `strategy` is
dropped) and sequential scans are switched off, so the planner has no choice but to use
HNSW. On a few hundred rows it would otherwise pick an exact scan and hide the behaviour
this file exists to guard: the per-strategy filter is applied after the HNSW scan.
"""
import os
from pathlib import Path

import pytest

URL = os.getenv("TEST_DATABASE_URL")
if not URL:
    pytest.skip("set TEST_DATABASE_URL to run the SQL integration tests", allow_module_level=True)

np = pytest.importorskip("numpy")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("pgvector")

TEST_SCHEMA = "_t_rl"
SCHEMA_SQL = Path(__file__).resolve().parent.parent / "sql" / "init.sql"
STRATEGIES = ["a", "b", "c"]
PER_STRATEGY = 300

os.environ["DATABASE_URL"] = URL  # before app.db builds its pool

from app import cache_pg  # noqa: E402
from app.db import close_pool, conn, open_pool  # noqa: E402
from app.ratelimit import allow_pg  # noqa: E402
from app.retrieval import dense_search_vec, lexical_search  # noqa: E402


@pytest.fixture(scope="module")
def vectors():
    with psycopg.connect(URL, autocommit=True) as c:
        # the extension must exist (in public) before the pool registers the vector type
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
        c.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
        c.execute(f"SET search_path = {TEST_SCHEMA}, public")
        c.execute(SCHEMA_SQL.read_text())
        c.execute(f"DROP INDEX {TEST_SCHEMA}.chunks_strategy_idx")

    # libpq applies PGOPTIONS to every new connection of this process
    os.environ["PGOPTIONS"] = f"-c search_path={TEST_SCHEMA},public -c enable_seqscan=off"
    open_pool()

    rng = np.random.default_rng(0)
    out = {}
    with conn() as c, c.cursor() as cur:
        for s in STRATEGIES:
            v = rng.normal(size=(PER_STRATEGY, 384)).astype("float32")
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            out[s] = v
            cur.executemany(
                "INSERT INTO documents (id) VALUES (%s)",
                [(f"{s}-{i}",) for i in range(PER_STRATEGY)],
            )
            cur.executemany(
                "INSERT INTO chunks (doc_id, strategy, idx, text, embedding) VALUES (%s, %s, 0, %s, %s)",
                [(f"{s}-{i}", s, f"chunk {i}", v[i]) for i in range(PER_STRATEGY)],
            )
        for doc_id, text in [
            ("v1", "Vitamin D supplementation reduces fracture risk in elderly adults"),
            ("v2", "Calcium intake and bone density in postmenopausal women"),
            ("v3", "Statin therapy lowers cardiovascular mortality"),
        ]:
            cur.execute("INSERT INTO documents (id) VALUES (%s)", (doc_id,))
            cur.execute(
                "INSERT INTO chunks (doc_id, strategy, idx, text) VALUES (%s, 'lex', 0, %s)",
                (doc_id, text),
            )
    yield out

    close_pool()
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")


def test_dense_returns_k_results_when_strategies_share_the_index(vectors):
    # Regression: without iterative scans the strategy filter is applied after the
    # HNSW scan and only a fraction of the requested rows come back.
    hits = dense_search_vec(vectors["a"][7], "a", 100)
    assert len(hits) == 100
    assert all(h.doc_id.startswith("a-") for h in hits)
    assert "a-7" in [h.doc_id for h in hits[:10]]


def test_dense_results_are_ordered_by_similarity(vectors):
    hits = dense_search_vec(vectors["b"][3], "b", 50)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_lexical_matches_natural_language_questions(vectors):
    hits = lexical_search("does vitamin D reduce fracture risk in older people", "lex", 5)
    assert hits and hits[0].doc_id == "v1"


def test_lexical_stopword_only_query_returns_nothing(vectors):
    assert lexical_search("is it the", "lex", 5) == []


def test_postgres_cache_hit_miss_and_scope(vectors):
    cache_pg.clear()
    q = vectors["a"][0]
    assert cache_pg.lookup(q, "recursive_256", "hybrid") is None

    cache_pg.store("what is x", q, "recursive_256", "hybrid", {"answer": "x", "sources": []})
    hit = cache_pg.lookup(q, "recursive_256", "hybrid")
    assert hit and hit["answer"] == "x" and hit["cache_similarity"] > 0.99

    # another retrieval configuration, or an unrelated question, must not hit
    assert cache_pg.lookup(q, "recursive_256", "dense") is None
    assert cache_pg.lookup(vectors["b"][1], "recursive_256", "hybrid") is None

    # storing the same question again replaces it instead of piling up
    cache_pg.store("what is x", q, "recursive_256", "hybrid", {"answer": "y", "sources": []})
    assert cache_pg.lookup(q, "recursive_256", "hybrid")["answer"] == "y"
    assert cache_pg.clear() == 1


def test_postgres_rate_limiter(vectors):
    results = [allow_pg("rl:test-ip", 3, now=1000) for _ in range(5)]
    assert results == [True, True, True, False, False]
    assert allow_pg("rl:other-ip", 3, now=1000)  # keys are independent
    assert allow_pg("rl:test-ip", 3, now=1060)  # the next window starts fresh
