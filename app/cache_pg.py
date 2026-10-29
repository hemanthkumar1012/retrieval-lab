"""Semantic cache in Postgres (used when REDIS_URL is not set).

Same idea as the Redis version, but the entries live next to the documents, so a serverless
deploy needs no second service. Lookup is a nearest-neighbour query over a small capped
table, filtered to one retrieval configuration and to entries younger than the TTL.
"""
from typing import Optional

import numpy as np
from psycopg.types.json import Jsonb

from .config import settings
from .db import conn

MAX_ENTRIES = 2000


def lookup(qvec: np.ndarray, strategy: str, mode: str) -> Optional[dict]:
    with conn() as c:
        row = c.execute(
            """SELECT payload, 1 - (embedding <=> %s) AS sim
               FROM semantic_cache
               WHERE scope = %s AND created_at > now() - make_interval(secs => %s)
               ORDER BY embedding <=> %s LIMIT 1""",
            (qvec, f"{strategy}|{mode}", settings.cache_ttl_seconds, qvec),
        ).fetchone()
    if row is None or float(row[1]) < settings.cache_threshold:
        return None
    out = dict(row[0])
    out["cache_similarity"] = round(float(row[1]), 4)
    return out


def store(query: str, qvec: np.ndarray, strategy: str, mode: str, payload: dict) -> None:
    scope = f"{strategy}|{mode}"
    with conn() as c:
        c.execute("DELETE FROM semantic_cache WHERE scope = %s AND query = %s", (scope, query))
        c.execute(
            "INSERT INTO semantic_cache (scope, query, embedding, payload) VALUES (%s, %s, %s, %s)",
            (scope, query, qvec, Jsonb(payload)),
        )
        # keep the table small: drop expired rows and anything beyond the newest MAX_ENTRIES
        c.execute(
            "DELETE FROM semantic_cache WHERE created_at < now() - make_interval(secs => %s)",
            (settings.cache_ttl_seconds,),
        )
        c.execute(
            "DELETE FROM semantic_cache WHERE id <= (SELECT max(id) FROM semantic_cache) - %s",
            (MAX_ENTRIES,),
        )


def clear() -> int:
    with conn() as c:
        return c.execute("DELETE FROM semantic_cache").rowcount
