import argparse
import json
import sys
from typing import Iterable

from .chunking import chunk
from .db import close_pool, conn, open_pool
from .embeddings import embed_docs


def ingest(docs: Iterable[dict], strategy: str, batch_docs: int = 64) -> int:
    """docs: dicts with id, text and optionally title/source. Returns chunk count."""
    total = 0
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) >= batch_docs:
            total += _flush(batch, strategy)
            batch = []
    if batch:
        total += _flush(batch, strategy)
    return total


def _flush(batch: list[dict], strategy: str) -> int:
    rows = []  # (doc_id, idx, text)
    for d in batch:
        body = d["text"]
        if d.get("title"):
            body = f"{d['title']}. {body}"
        for i, c in enumerate(chunk(body, strategy)):
            rows.append((d["id"], i, c))
    if not rows:
        return 0

    vecs = embed_docs([r[2] for r in rows])

    with conn() as c:
        with c.cursor() as cur:
            cur.executemany(
                """INSERT INTO documents (id, title, source) VALUES (%s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, source = EXCLUDED.source""",
                [(d["id"], d.get("title"), d.get("source")) for d in batch],
            )
            ids = [d["id"] for d in batch]
            cur.execute(
                "DELETE FROM chunks WHERE strategy = %s AND doc_id = ANY(%s)", (strategy, ids)
            )
            cur.executemany(
                """INSERT INTO chunks (doc_id, strategy, idx, text, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                [(doc_id, strategy, idx, text, vec) for (doc_id, idx, text), vec in zip(rows, vecs)],
            )
        c.commit()
    return len(rows)


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="ingest a jsonl file of {id, title?, text}")
    ap.add_argument("path")
    ap.add_argument("--strategy", default="recursive_256")
    args = ap.parse_args()

    open_pool()
    try:
        n = ingest(_read_jsonl(args.path), args.strategy)
        print(f"{n} chunks written with strategy={args.strategy}", file=sys.stderr)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
