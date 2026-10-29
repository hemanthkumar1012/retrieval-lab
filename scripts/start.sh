#!/bin/sh
# Container entrypoint.
# With no REDIS_URL set (a single-container host such as a Hugging Face Space) it starts a
# throwaway Redis beside the API. The cache and rate-limit counters are rebuildable, so
# losing them on restart is fine. With REDIS_URL set (docker compose, Upstash, ...) it is
# left alone and nothing extra is started.
set -e
if [ -z "${REDIS_URL:-}" ]; then
  redis-server --daemonize yes --bind 127.0.0.1 --save "" --appendonly no \
    --maxmemory 64mb --maxmemory-policy allkeys-lru >/dev/null
  export REDIS_URL=redis://127.0.0.1:6379/0
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
