"""Front door for the semantic cache and the rate-limit counters.

Redis when REDIS_URL is set (docker compose, local development). Otherwise Postgres, the
same Supabase database that holds the documents, which is what a serverless deploy uses.
"""
from .config import settings

if settings.redis_url:
    from . import cache_redis as _backend
    from .ratelimit import allow as _allow

    BACKEND = "redis"

    def rate_limit_allow(key: str, limit: int) -> bool:
        return _allow(_backend.client(), key, limit)
else:
    from . import cache_pg as _backend
    from .ratelimit import allow_pg

    BACKEND = "postgres"

    def rate_limit_allow(key: str, limit: int) -> bool:
        return allow_pg(key, limit)

lookup = _backend.lookup
store = _backend.store
clear = _backend.clear
