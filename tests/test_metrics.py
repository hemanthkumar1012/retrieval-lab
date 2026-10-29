import math

from bench.metrics import dedupe_docs, mrr_at_k, ndcg_at_k, percentile, recall_at_k

REL = {"a": 1, "b": 1}


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe_docs(["x", "a", "x", "b", "a"]) == ["x", "a", "b"]


def test_recall():
    assert recall_at_k(["x", "a", "b"], REL, 1) == 0.0
    assert recall_at_k(["x", "a", "b"], REL, 2) == 0.5
    assert recall_at_k(["x", "a", "b"], REL, 3) == 1.0
    assert recall_at_k(["a"], {}, 5) == 0.0


def test_mrr():
    assert mrr_at_k(["x", "y", "a"], REL, 10) == 1 / 3
    assert mrr_at_k(["x", "y", "a"], REL, 2) == 0.0


def test_ndcg_perfect_and_worst():
    assert math.isclose(ndcg_at_k(["a", "b"], REL, 10), 1.0)
    assert ndcg_at_k(["x", "y"], REL, 10) == 0.0


def test_ndcg_known_value():
    # one relevant doc at rank 2 out of two relevant: dcg = 1/log2(3), idcg = 1 + 1/log2(3)
    got = ndcg_at_k(["x", "a"], REL, 10)
    want = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(got, want)


def test_percentile():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50
    assert percentile(vals, 95) == 95
    assert percentile([], 50) == 0.0
