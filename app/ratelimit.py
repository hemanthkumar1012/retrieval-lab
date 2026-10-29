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
