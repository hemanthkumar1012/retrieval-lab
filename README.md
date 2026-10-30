# retrieval-lab

[![ci](https://github.com/hemanthkumar1012/retrieval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hemanthkumar1012/retrieval-lab/actions/workflows/ci.yml)

A retrieval-augmented generation service where the retrieval side is measured, not assumed.
Hybrid search over Postgres (pgvector + full-text), cross-encoder reranking, a semantic
cache, and streamed answers over SSE. Benchmark scripts compare chunking strategies and
retrieval modes on BEIR SciFact and help pick a safe cache threshold.

**Live demo: https://retrieval-lab1.vercel.app** (5,183 SciFact abstracts). Ask a question, watch
each stage's time, then open *Compare modes* to see how meaning search, keyword search, fusion
and reranking order the same results.

## Stack

Python 3.12, FastAPI, Postgres + pgvector 0.8 (HNSW) on Supabase, ONNX Runtime with int8
models (no PyTorch at runtime), deployed as one Vercel function. The semantic cache and the
rate limiter live in the same Postgres database; with `REDIS_URL` set they use Redis instead
(that is what `docker compose` does). The front end is plain HTML, CSS and JavaScript served by
the API, no build step.

## How retrieval works

1. **Dense**: `bge-small-en-v1.5` embeddings (int8 ONNX), cosine distance through an HNSW index.
2. **Lexical**: Postgres full-text search, query terms OR-ed together and ranked with
   `ts_rank_cd`. This is BM25-like but not literally BM25 (no IDF term in `ts_rank_cd`).
3. **Fusion**: reciprocal rank fusion over the two candidate lists (k=60).
4. **Rerank**: `ms-marco-MiniLM-L-6-v2` cross-encoder (int8 ONNX) over the fused top 20.

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

The models ship in `models/` as int8 ONNX files (56 MB). `scripts/export_models.py` rebuilds them
from the original weights and writes `models/PARITY.md`, which compares them with the PyTorch
versions. Run it from the **export-models** workflow, not on every push.

## Benchmarks

```bash
make bench          # all strategies x all retrieval modes -> bench/results/scifact.md
make cache-bench    # threshold sweep for the semantic cache -> bench/results/cache.md
```

Reported per configuration: recall@1/5/10, MRR@10, nDCG@10 and p50/p95 latency
(query embedding included). Chunk hits are collapsed to documents before scoring,
because SciFact labels are per document. To get enough chunks for that, the reranked mode
reranks the top 40 fused chunks here while the API reranks 20, so its latency is a little higher.

### Results

_Not filled in yet. Run `make bench` and paste `bench/results/scifact.md` here, or use the
**bench** workflow (Actions tab, Run workflow), which runs it on a GitHub runner and prints
both tables in the job summary. Latency depends on the machine, so say which one you used._

## Semantic cache

Query embeddings are compared against recent cached queries (same strategy and mode).
Above `CACHE_THRESHOLD` cosine similarity the stored answer is replayed. In Postgres the lookup
is a pgvector distance query over at most 2000 entries; the Redis backend scans them in Python.
`make cache-bench` shows the trade-off: how many rephrasings hit, versus how often a
different question wrongly hits. Pick the threshold from that table, not by feel.

## Development

```bash
pip install -r requirements-dev.txt
make lint
make test
```

Unit tests cover rank fusion, the evaluation metrics, the chunkers, the rate limiter and the
dependency lists, and need no database. `make smoke` ingests three documents into a running API and checks every
retrieval mode and the streaming endpoint (it adds three `smoke-*` documents to the
`fixed_128` strategy). `make test-sql` runs the retrieval SQL against a real pgvector (CI does the
same in a service container). CI runs lint, unit tests, the SQL tests, and the smoke test against
the full docker compose stack.

## Deploy

Supabase holds the data and Vercel runs the API. Both free plans are enough. Step by step, with
the environment variables, in [docs/deploy.md](docs/deploy.md). Set `ADMIN_TOKEN` on anything
public. Pushing to `main` redeploys, and the free Vercel plan limits how many deployments a day
you get, so batch changes into fewer pushes.

## Layout

```
app/        service: chunking, embeddings, fusion, retrieval, cache, llm, api
app/static/ the web page (index.html, styles.css, app.js)
models/     int8 ONNX embedder and reranker, plus the parity report
bench/      metrics, retrieval benchmark, cache threshold benchmark, SciFact loader
scripts/    model export, smoke test
tests/      unit and SQL tests
sql/        schema
docs/       architecture notes, deploy guide
```

## License

MIT
