import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import cache
from .chunking import STRATEGIES
from .config import settings
from .db import close_pool, conn, open_pool
from .embeddings import embed_query
from .ingest import ingest
from .llm import LLMError, build_messages, stream_completion
from .retrieval import MODES, retrieve

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.admin_token:
        log.warning("ADMIN_TOKEN is not set: /ingest and DELETE /cache are open to anyone")
    open_pool()
    # load models now so the first request doesn't pay for it
    await run_in_threadpool(embed_query, "warmup")
    yield
    close_pool()


app = FastAPI(title="retrieval-lab", lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Guards the endpoints that write. Open when ADMIN_TOKEN is unset (local dev)."""
    if not settings.admin_token:
        return
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(401, "missing or wrong X-Admin-Token header")


def rate_limit(request: Request):
    """Per-IP limit on the endpoints that cost CPU or LLM tokens."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    if not cache.rate_limit_allow(f"rl:{ip}", limit):
        raise HTTPException(429, "rate limit exceeded, try again in a minute")


class Doc(BaseModel):
    id: str
    text: str
    title: str | None = None
    source: str | None = None


class IngestRequest(BaseModel):
    docs: list[Doc]
    strategy: str = settings.default_strategy


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)
    mode: str = "hybrid_rerank"
    strategy: str = settings.default_strategy
    use_cache: bool = True


def _check(req: QueryRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {sorted(STRATEGIES)}")


def _hit_json(h):
    return {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 5),
            "ranks": h.ranks, "text": h.text}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    with conn() as c:
        n = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"ok": True, "chunks": n, "cache": cache.BACKEND}


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest_docs(req: IngestRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy {req.strategy!r}")
    n = ingest([d.model_dump() for d in req.docs], req.strategy)
    return {"docs": len(req.docs), "chunks": n, "strategy": req.strategy}


@app.post("/search", dependencies=[Depends(rate_limit)])
def search(req: QueryRequest):
    _check(req)
    hits, timings = retrieve(req.query, req.strategy, req.mode, req.k)
    return {"hits": [_hit_json(h) for h in hits], "timings_ms": {k: round(v, 1) for k, v in timings.items()}}


@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(req: QueryRequest):
    """Server-sent events: `meta` first (sources, timings, cache info), then
    `token` events as the model writes, then `done`."""
    _check(req)

    async def gen():
        t0 = time.perf_counter()
        qvec = await run_in_threadpool(embed_query, req.query)

        if req.use_cache:
            hit = await run_in_threadpool(cache.lookup, qvec, req.strategy, req.mode)
            if hit:
                yield _sse("meta", {"cache": "hit", "similarity": hit["cache_similarity"],
                                    "sources": hit["sources"]})
                yield _sse("token", hit["answer"])
                yield _sse("done", {"total_ms": round((time.perf_counter() - t0) * 1000, 1)})
                return

        hits, timings = await run_in_threadpool(retrieve, req.query, req.strategy, req.mode, req.k)
        sources = [_hit_json(h) for h in hits]
        yield _sse("meta", {"cache": "miss", "sources": sources,
                            "timings_ms": {k: round(v, 1) for k, v in timings.items()}})

        parts = []
        first = None
        try:
            async for delta in stream_completion(build_messages(req.query, hits)):
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                parts.append(delta)
                yield _sse("token", delta)
        except LLMError as e:
            yield _sse("error", {"message": str(e)})
            return
        except httpx.HTTPError as e:
            # sources were already sent; tell the client instead of dropping the stream
            status = getattr(getattr(e, "response", None), "status_code", None)
            yield _sse("error", {"message": f"LLM request failed ({status or type(e).__name__})"})
            return

        answer = "".join(parts)
        if req.use_cache and answer and settings.llm_api_key:
            await run_in_threadpool(
                cache.store, req.query, qvec, req.strategy, req.mode,
                {"answer": answer, "sources": sources},
            )
        yield _sse("done", {"total_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "first_token_ms": round(first, 1) if first else None})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.delete("/cache", dependencies=[Depends(require_admin)])
def clear_cache():
    return {"cleared": cache.clear()}
