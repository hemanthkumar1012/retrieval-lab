# retrieval-lab

[![ci](https://github.com/hemanthkumar1012/retrieval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hemanthkumar1012/retrieval-lab/actions/workflows/ci.yml)

A retrieval-augmented generation service where the retrieval side is measured, not assumed.
Hybrid search over Postgres (pgvector + full-text), cross-encoder reranking, a Redis
semantic cache, and streamed answers over SSE. Benchmark scripts compare chunking
strategies and retrieval modes on BEIR SciFact and help pick a safe cache threshold.

## Stack

Python 3.11, FastAPI, Postgres 16 + pgvector (HNSW), Redis, sentence-transformers, Docker.

## How retrieval works

1. **Dense**: `bge-small-en-v1.5` embeddings, cosine distance through an HNSW index.
2. **Lexical**: Postgres full-text search, query terms OR-ed together and ranked with
   `ts_rank_cd`. This is BM25-like but not literally BM25 (no IDF term in `ts_rank_cd`).
3. **Fusion**: reciprocal rank fusion over the two candidate lists (k=60).
4. **Rerank**: `ms-marco-MiniLM-L-6-v2` cross-encoder over the fused top candidates.

Chunking strategies (`fixed_128/256/512`, `sentence_256`, `recursive_256/512`) are stored
side by side in the `chunks` table, keyed by a `strategy` column, so they can be compared
on identical data. See [docs/architecture.md](docs/architecture.md) for the design notes.

## Quick start

```bash
cp .env.example .env          # set LLM_API_KEY if you want generated answers
make up                       # postgres + redis
pip install -r requirements.txt
make bench-quick              # ~100 queries, two chunking strategies
```

Then run the API and open http://localhost:8000 for a small streaming demo page
(ingest a corpus first with `python -m app.ingest docs.jsonl`, or run the benchmark, which ingests SciFact):

```bash
make run
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "does vitamin D reduce fracture risk", "mode": "hybrid_rerank"}'
curl -N -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query": "does vitamin D reduce fracture risk"}'
```

`/ask` streams `meta` (sources, per-stage timings, cache hit or miss), then `token`
events, then `done`. Any OpenAI-compatible endpoint works through `LLM_BASE_URL`.
Everything also runs in Docker: `docker compose up --build`.

## Benchmarks

```bash
make bench          # all strategies x all retrieval modes -> bench/results/scifact.md
make cache-bench    # threshold sweep for the semantic cache -> bench/results/cache.md
```

Reported per configuration: recall@1/5/10, MRR@10, nDCG@10 and p50/p95 latency
(query embedding included). Chunk hits are collapsed to documents before scoring,
because SciFact labels are per document. The reranked modes rerank the top 100 fused
candidates here (the API default is 50), so latency reflects that.

### Results

_Not filled in yet. Run `make bench` and paste `bench/results/scifact.md` here, or use the
**bench** workflow (Actions tab, Run workflow), which runs it on a GitHub runner and prints
both tables in the job summary. Latency depends on the machine, so say which one you used._

## Semantic cache

Query embeddings are compared against recent cached queries (same strategy and mode).
Above `CACHE_THRESHOLD` cosine similarity the stored answer is replayed. Lookup is a
linear scan over at most 2000 entries, fine at this scale (noted in `app/cache.py`).
`make cache-bench` shows the trade-off: how many rephrasings hit, versus how often a
different question wrongly hits. Pick the threshold from that table, not by feel.

## Development

```bash
pip install -r requirements-dev.txt
make lint
make test
```

Unit tests cover rank fusion, the evaluation metrics, the chunkers and the rate limiter, and
need no database. `make smoke` ingests three documents into a running API and checks every
retrieval mode and the streaming endpoint (it adds three `smoke-*` documents to the
`fixed_128` strategy). `make test-sql` runs the retrieval SQL against a real pgvector (CI does the
same in a service container). CI runs lint, unit tests, the SQL tests, and the smoke test against
the full docker compose stack.

## Deploy

Supabase for Postgres + pgvector, Upstash for Redis, any Docker host for the API. Step by
step, with the environment variables, in [docs/deploy.md](docs/deploy.md). `ADMIN_TOKEN`
should be set on anything public.

## Layout

```
app/        service: chunking, embeddings, fusion, retrieval, cache, llm, api, demo page
bench/      metrics, retrieval benchmark, cache threshold benchmark
tests/      unit tests
sql/        schema
docs/       architecture notes, deploy guide
```

## License

MIT
