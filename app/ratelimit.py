"""Fixed-window rate limiter on Redis INCR. No dependencies beyond a redis-like client."""
import time


def allow(r, key: str, limit: int, window: int = 60, now: float | None = None) -> bool:
    """True if this call is within `limit` calls per `window` seconds for `key`.

    Fails open: if Redis is unreachable the request is let through rather than
    taking the whole API down with it.
    """
    bucket = int((time.time() if now is None else now) // window)
    k = f"{key}:{bucket}"
    try:
        n = r.incr(k)
        if n == 1:
            r.expire(k, window * 2)
    except Exception:
        return True
    return n <= limit


def allow_pg(key: str, limit: int, window: int = 60, now: float | None = None) -> bool:
    """Same fixed window as `allow`, counted in Postgres. Fails open on database errors."""
    from .db import conn  # imported late so this module stays importable without psycopg

    bucket = int((time.time() if now is None else now) // window)
    try:
        with conn() as c:
            n = c.execute(
                """INSERT INTO rate_limits (key, bucket, n) VALUES (%s, %s, 1)
                   ON CONFLICT (key, bucket) DO UPDATE SET n = rate_limits.n + 1
                   RETURNING n""",
                (key, bucket),
            ).fetchone()[0]
            if n == 1:  # first hit of a new window: forget old ones
                c.execute("DELETE FROM rate_limits WHERE bucket < %s", (bucket - 2,))
    except Exception:
        return True
    return n <= limit
