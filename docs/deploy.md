# Deploying

This is the setup behind the live demo: Supabase for the data, Vercel for the API and the page.

```
browser ──> Vercel function (FastAPI + two ONNX models) ──> Supabase Postgres
                                                              documents, chunks (pgvector)
                                                              semantic cache, rate limits
```

The models are int8 ONNX files in `models/`, so the function needs no PyTorch and the bundle stays
well under Vercel's 500 MB limit (CI fails if it grows past 400 MB). The cache and the rate
limiter use Postgres, so there is nothing else to host.

## 1. Database (Supabase)

1. Create a project. In **Database → Extensions** enable `vector` (pgvector 0.8 or newer is
   needed for iterative index scans).
2. Open the **SQL editor**, paste `sql/init.sql`, run it. It also turns on row level security,
   so the public REST API and anon key cannot read or write these tables.
3. **Connect** → copy a **pooler** string. Use the **Session pooler** (port 5432) for loading
   data from GitHub Actions or your machine, and the **Transaction pooler** (port 6543) for
   Vercel. The direct connection is IPv6-only, which GitHub runners and many hosts cannot reach.
   The app disables prepared statements, so both poolers work.

## 2. Load data

The **load-data** workflow loads the SciFact corpus with one chunking strategy:

1. Repo **Settings → Secrets and variables → Actions → New repository secret**: name
   `DATABASE_URL`, value the session pooler string with your password filled in.
2. **Actions → load-data → Run workflow**. It takes about ten minutes and can be re-run, it skips
   what is already stored.

Load **one** strategy for the demo. The benchmark's six strategies store the corpus six times and
the free database is limited to 500 MB. `recursive_256` is 69 MB. To load your own documents,
POST them to `/ingest` with the `X-Admin-Token` header, or run `python -m app.ingest docs.jsonl`.

## 3. API (Vercel)

1. **Add New → Project**, import the GitHub repo. Leave the framework preset and root directory
   alone. Vercel detects FastAPI from `app/main.py`.
2. Before the first deploy, add these environment variables (mark the secrets **Sensitive**):

| variable | value |
|---|---|
| `DATABASE_URL` | Supabase transaction pooler string (port 6543) |
| `ADMIN_TOKEN` | long random string, required on `/ingest` and `DELETE /cache` |
| `DB_POOL_MAX` | `3`, every function instance opens its own pool |
| `RATE_LIMIT_PER_MINUTE` | `30` (per IP, on `/search` and `/ask`) |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | your provider. Leave the key empty for retrieval-only |

Leave `REDIS_URL` unset. That is what switches the cache and the rate limiter to Postgres.

Vercel installs the packages listed in `pyproject.toml`, not `requirements.txt`. The Docker image
uses `requirements.txt`, and a test checks that the two lists stay equal.

Any OpenAI-compatible chat endpoint works. For Gemini use
`LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai` and a current model name,
since Google retires old ones for new keys. A wrong name shows up on the page as the provider's
own error message.

Set a spending cap with your LLM provider as well. The rate limit is per IP and best effort.

## 4. Check it

```bash
curl https://<your-app>.vercel.app/health
curl -X POST https://<your-app>.vercel.app/search -H 'content-type: application/json' \
  -d '{"query": "does vitamin D reduce fracture risk", "mode": "hybrid_rerank"}'
curl -X DELETE https://<your-app>.vercel.app/cache                                  # expect 401
curl -X DELETE https://<your-app>.vercel.app/cache -H "X-Admin-Token: $ADMIN_TOKEN"  # expect 200
```

`/health` returns the chunk count and which cache backend is active (`postgres` here), so a count
of `0` means the loader pointed at a different database.

If a request fails, **Logs** in the Vercel dashboard shows the Python traceback. The first request
after a quiet period is slower while the function loads the models.

## Running it somewhere else

`docker compose up --build` starts Postgres, Redis and the API on a laptop or any Docker host.
The image honours `PORT`, so Render, Railway or Fly.io work too. Set `REDIS_URL` to share one
cache across several instances.
