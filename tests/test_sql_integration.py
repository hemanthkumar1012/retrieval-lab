"""Runs the retrieval SQL against a real Postgres with pgvector.

    TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest tests/test_sql_integration.py

Skipped when TEST_DATABASE_URL is not set. The test only touches rows whose strategy
starts with `_t_` and whose document id starts with `_t_`, and removes them afterwards.
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

os.environ["DATABASE_URL"] = URL  # before app.db builds its pool
# On a few hundred rows Postgres prefers an exact sequential scan, which hides the HNSW
# behaviour under test. libpq applies PGOPTIONS to every new connection of this process only.
os.environ["PGOPTIONS"] = "-c enable_seqscan=off"

from app.db import close_pool, conn, open_pool  # noqa: E402
from app.retrieval import dense_search_vec, lexical_search  # noqa: E402

SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "init.sql"
STRATEGIES = ["_t_a", "_t_b", "_t_c"]
PER_STRATEGY = 300


def _cleanup(c):
    c.execute("DELETE FROM chunks WHERE strategy LIKE '\\_t\\_%'")
    c.execute("DELETE FROM documents WHERE id LIKE '\\_t\\_%'")


@pytest.fixture(scope="module")
def vectors():
    # the extension must exist before the pool registers the vector type
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(SCHEMA.read_text())

    open_pool()
    rng = np.random.default_rng(0)
    out = {}
    with conn() as c, c.cursor() as cur:
        _cleanup(c)
        for s in STRATEGIES:
            v = rng.normal(size=(PER_STRATEGY, 384)).astype("float32")
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            out[s] = v
            cur.executemany(
                "INSERT INTO documents (id) VALUES (%s) ON CONFLICT DO NOTHING",
                [(f"{s}-{i}",) for i in range(PER_STRATEGY)],
            )
            cur.executemany(
                "INSERT INTO chunks (doc_id, strategy, idx, text, embedding) VALUES (%s, %s, 0, %s, %s)",
                [(f"{s}-{i}", s, f"chunk {i}", v[i]) for i in range(PER_STRATEGY)],
            )
        for doc_id, text in [
            ("_t_v1", "Vitamin D supplementation reduces fracture risk in elderly adults"),
            ("_t_v2", "Calcium intake and bone density in postmenopausal women"),
            ("_t_v3", "Statin therapy lowers cardiovascular mortality"),
        ]:
            c.execute("INSERT INTO documents (id) VALUES (%s) ON CONFLICT DO NOTHING", (doc_id,))
            c.execute(
                "INSERT INTO chunks (doc_id, strategy, idx, text) VALUES (%s, '_t_lex', 0, %s)",
                (doc_id, text),
            )
    yield out
    with conn() as c:
        _cleanup(c)
    close_pool()


def test_dense_returns_k_results_when_strategies_share_the_index(vectors):
    # Regression: without iterative scans the strategy filter is applied after the
    # HNSW scan and only a fraction of the requested rows come back.
    hits = dense_search_vec(vectors["_t_a"][7], "_t_a", 100)
    assert len(hits) == 100
    assert all(h.doc_id.startswith("_t_a-") for h in hits)
    assert "_t_a-7" in [h.doc_id for h in hits[:10]]


def test_dense_results_are_ordered_by_similarity(vectors):
    hits = dense_search_vec(vectors["_t_b"][3], "_t_b", 50)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_lexical_matches_natural_language_questions(vectors):
    hits = lexical_search("does vitamin D reduce fracture risk in older people", "_t_lex", 5)
    assert hits and hits[0].doc_id == "_t_v1"


def test_lexical_stopword_only_query_returns_nothing(vectors):
    assert lexical_search("is it the", "_t_lex", 5) == []
