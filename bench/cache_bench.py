"""How should CACHE_THRESHOLD be set?

A semantic cache has two ways to be wrong: a miss on a question that was
already answered (wasted work), and a hit on a different question (wrong
answer served). This script embeds SciFact test queries, builds rephrased
variants of each, and sweeps the cosine threshold to show both rates.

    python -m bench.cache_bench

No database or Redis needed, only the embedding model.

The rephrasings are mechanical templates, so they are easier than real user
paraphrases. Treat hit rate as an upper bound and the false-hit rate as the
number to care about.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tabulate import tabulate

from app.embeddings import embed_docs

RESULTS = Path(__file__).parent / "results"

TEMPLATES = [
    lambda q: q.rstrip(".?!") + "?",
    lambda q: "Is it true that " + q[0].lower() + q[1:].rstrip(".?!") + "?",
    lambda q: "Can you tell me whether " + q[0].lower() + q[1:].rstrip(".?!") + "?",
    lambda q: q.lower().rstrip(".?!"),
]


def load_queries(limit: int | None) -> list[str]:
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")
    test_ids = {str(r["query-id"]) for r in qrels}
    out = [q["text"] for q in queries if str(q["_id"]) in test_ids]
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qs = load_queries(args.limit)
    originals = embed_docs(qs)  # (n, d), L2-normalised

    variants, owner = [], []
    for i, q in enumerate(qs):
        for t in TEMPLATES:
            variants.append(t(q))
            owner.append(i)
    var_emb = embed_docs(variants)
    owner = np.array(owner)

    # paraphrase hit: best match in the cache is the query it was derived from
    sims = var_emb @ originals.T
    own_sim = sims[np.arange(len(variants)), owner]
    sims_other = sims.copy()
    sims_other[np.arange(len(variants)), owner] = -1
    paraphrase_best_other = sims_other.max(axis=1)

    # false hit: a distinct query finds some other query in the cache
    q2q = originals @ originals.T
    np.fill_diagonal(q2q, -1)
    distinct_best = q2q.max(axis=1)

    rows = []
    for thr in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        wrong_hit = (paraphrase_best_other >= thr) & (paraphrase_best_other > own_sim)
        rows.append({
            "threshold": thr,
            "paraphrase_hit_rate": float((own_sim >= thr).mean()),
            "paraphrase_wrong_hit_rate": float(wrong_hit.mean()),
            "distinct_false_hit_rate": float((distinct_best >= thr).mean()),
        })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cache.json").write_text(json.dumps(rows, indent=2))
    table = tabulate(
        [[r["threshold"], f"{r['paraphrase_hit_rate']:.1%}", f"{r['paraphrase_wrong_hit_rate']:.1%}",
          f"{r['distinct_false_hit_rate']:.1%}"] for r in rows],
        headers=["threshold", "paraphrase hit", "paraphrase wrong-hit", "distinct-query false hit"],
        tablefmt="github",
    )
    (RESULTS / "cache.md").write_text(table + "\n")
    print(f"{len(qs)} queries, {len(variants)} variants\n")
    print(table)


if __name__ == "__main__":
    main()
