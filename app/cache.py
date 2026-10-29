"""Semantic cache on plain Redis.

Each entry is a hash holding the query embedding (float32 bytes) and the
answer. Lookup is a linear scan over a small capped set of recent entries,
which is fine up to a few thousand entries and avoids needing RediSearch.
If this ever needs to scale, swap in an HNSW index; the interface stays the same.
"""
import hashlib
import json
import time
from typing import Optional

import numpy as np
import redis

from .config import settings

_r = redis.Redis.from_url(settings.redis_url)

INDEX_KEY = "semcache:index"
MAX_ENTRIES = 2000


def _key(strategy: str, mode: str, q: str) -> str:
    h = hashlib.sha1(f"{strategy}|{mode}|{q}".encode()).hexdigest()
    return f"semcache:{h}"


def lookup(qvec: np.ndarray, strategy: str, mode: str) -> Optional[dict]:
    keys = _r.zrevrange(INDEX_KEY, 0, MAX_ENTRIES - 1)
    if not keys:
        return None

    pipe = _r.pipeline()
    for k in keys:
        pipe.hmget(k, "vec", "scope")
    rows = pipe.execute()

    scope = f"{strategy}|{mode}".encode()
    best, best_key = -1.0, None
    for k, (raw, sc) in zip(keys, rows, strict=True):
        if raw is None or sc != scope:
            continue  # expired or from a different retrieval config
        v = np.frombuffer(raw, dtype=np.float32)
        sim = float(np.dot(qvec, v))  # both L2-normalised
        if sim > best:
            best, best_key = sim, k

    if best_key is None or best < settings.cache_threshold:
        return None

    payload = _r.hget(best_key, "payload")
    if payload is None:
        return None
    out = json.loads(payload)
    out["cache_similarity"] = round(best, 4)
    return out


def store(query: str, qvec: np.ndarray, strategy: str, mode: str, payload: dict) -> None:
    key = _key(strategy, mode, query)
    pipe = _r.pipeline()
    pipe.hset(
        key,
        mapping={
            "vec": qvec.astype(np.float32).tobytes(),
            "scope": f"{strategy}|{mode}",
            "payload": json.dumps(payload),
        },
    )
    pipe.expire(key, settings.cache_ttl_seconds)
    pipe.zadd(INDEX_KEY, {key: time.time()})
    pipe.zremrangebyrank(INDEX_KEY, 0, -(MAX_ENTRIES + 1))
    pipe.execute()


def clear() -> int:
    keys = _r.zrange(INDEX_KEY, 0, -1)
    if keys:
        _r.delete(*keys)
    _r.delete(INDEX_KEY)
    return len(keys)


def client() -> redis.Redis:
    """Shared connection, also used by the rate limiter."""
    return _r
