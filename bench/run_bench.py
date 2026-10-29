"""Retrieval benchmark on BEIR SciFact.

    python -m bench.run_bench
    python -m bench.run_bench --strategies recursive_256 fixed_256 --modes dense hybrid_rerank
    python -m bench.run_bench --limit 100        # quick smoke run

Ingests the corpus once per chunking strategy (skipped if already present),
runs every test query through every retrieval mode, and writes a markdown
table plus raw json into bench/results/.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tabulate import tabulate

from app.chunking import STRATEGIES
from app.db import close_pool, conn, open_pool
from app.ingest import ingest
from app.retrieval import MODES, retrieve

from .metrics import dedupe_docs, mrr_at_k, ndcg_at_k, percentile, recall_at_k

RESULTS = Path(__file__).parent / "results"
CHUNKS_PER_QUERY = 40  # enough chunks that doc-level dedupe still leaves 10 docs


def load_scifact(limit: int | None):
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    rel: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qrels:
        if r["score"] > 0:
            rel[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])

    qtext = {str(q["_id"]): q["text"] for q in queries}
    test_qs = [(qid, qtext[qid]) for qid in rel if qid in qtext]
    if limit:
        test_qs = test_qs[:limit]

    docs = ({"id": str(d["_id"]), "title": d["title"], "text": d["text"]} for d in corpus)
    return docs, test_qs, rel, len(corpus)


def scifact_docs_present(strategy: str, ids: list[str]) -> tuple[int, int]:
    """(SciFact docs stored for this strategy, all docs stored for this strategy)."""
    with conn() as c:
        have = c.execute(
            "SELECT count(DISTINCT doc_id) FROM chunks WHERE strategy = %s AND doc_id = ANY(%s)",
            (strategy, ids),
        ).fetchone()[0]
        total = c.execute(
            "SELECT count(DISTINCT doc_id) FROM chunks WHERE strategy = %s", (strategy,)
        ).fetchone()[0]
    return have, total


def run_config(strategy: str, mode: str, queries, rel, ks=(1, 5, 10)):
    lat, stage = [], defaultdict(list)
    r = {k: [] for k in ks}
    mrr, ndcg = [], []

    for qid, text in queries:
        hits, timings = retrieve(text, strategy, mode, k=CHUNKS_PER_QUERY, candidates=100)
        lat.append(timings["total"])
        for name, ms in timings.items():
            stage[name].append(ms)

        ranked = dedupe_docs([h.doc_id for h in hits])
        for k in ks:
            r[k].append(recall_at_k(ranked, rel[qid], k))
        mrr.append(mrr_at_k(ranked, rel[qid], 10))
        ndcg.append(ndcg_at_k(ranked, rel[qid], 10))

    row = {
        "strategy": strategy,
        "mode": mode,
        **{f"recall@{k}": statistics.mean(r[k]) for k in ks},
        "mrr@10": statistics.mean(mrr),
        "ndcg@10": statistics.mean(ndcg),
        "p50_ms": percentile(lat, 50),
        "p95_ms": percentile(lat, 95),
        "stage_p50_ms": {n: percentile(v, 50) for n, v in stage.items() if n != "total"},
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N test queries")
    args = ap.parse_args()

    docs, queries, rel, n_docs = load_scifact(args.limit)
    docs = list(docs)
    ids = [d["id"] for d in docs]
    print(f"SciFact: {n_docs} docs, {len(queries)} test queries")

    open_pool()
    try:
        for s in args.strategies:
            have, total = scifact_docs_present(s, ids)
            if have == len(ids):
                print(f"[{s}] SciFact already ingested, skipping")
            else:
                t = time.time()
                n = ingest(iter(docs), s)
                print(f"[{s}] {n} chunks in {time.time() - t:.0f}s")
                have, total = scifact_docs_present(s, ids)
            if total > have:
                print(f"[{s}] warning: {total - have} documents outside SciFact are also stored for "
                      "this strategy and act as extra distractors, so numbers are not comparable")

        # one throwaway query per mode so model load time doesn't land in p95
        for m in args.modes:
            retrieve("warmup", args.strategies[0], m, k=5)

        rows = []
        for s in args.strategies:
            for m in args.modes:
                row = run_config(s, m, queries, rel)
                rows.append(row)
                print(f"  {s:14s} {m:14s} recall@10={row['recall@10']:.3f} p50={row['p50_ms']:.0f}ms")
    finally:
        close_pool()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scifact.json").write_text(json.dumps(rows, indent=2))

    table = tabulate(
        [
            [r["strategy"], r["mode"], f"{r['recall@1']:.3f}", f"{r['recall@5']:.3f}",
             f"{r['recall@10']:.3f}", f"{r['mrr@10']:.3f}", f"{r['ndcg@10']:.3f}",
             f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}"]
            for r in rows
        ],
        headers=["chunking", "retrieval", "R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "p50 ms", "p95 ms"],
        tablefmt="github",
    )
    (RESULTS / "scifact.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
