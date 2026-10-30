"""Load the BEIR SciFact corpus into whatever database DATABASE_URL points at.

    python -m bench.load_scifact --strategy recursive_256

Safe to re-run: documents that are already stored for the strategy are skipped.
"""

import argparse

from app.chunking import STRATEGIES
from app.db import close_pool, open_pool
from app.ingest import ingest

from .run_bench import load_scifact, scifact_docs_present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="recursive_256", choices=sorted(STRATEGIES))
    args = ap.parse_args()

    open_pool()
    try:
        docs, _, _, n = load_scifact(None)
        docs = list(docs)
        have, total = scifact_docs_present(args.strategy, [d["id"] for d in docs])
        print(f"{have}/{n} SciFact docs already stored for {args.strategy} ({total} docs in total)")
        if have >= n:
            return
        chunks = ingest(docs, args.strategy)
        print(f"ingested {n} docs, {chunks} chunks")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
