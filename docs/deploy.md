# Deploying

The code is ready to deploy. Nothing here has been run against Supabase, Upstash or a
host yet, so treat the first deploy as the test and check each step below.

```
browser ──> API container (FastAPI + 2 models) ──> Supabase Postgres (pgvector)
                         └────────────────────────> Upstash Redis (semantic cache)
```

**Where Vercel fits.** Not for the API. It loads two PyTorch models at startup and keeps a
connection pool, which suits a long-running container rather than a per-request function.
The API already serves the demo page at `/`, so a separate frontend is optional. If you do
put one on Vercel, set `CORS_ORIGINS` to its URL.

## 1. Database (Supabase)

1. Create a project. In **Database → Extensions** enable `vector`.
2. Open the **SQL editor**, paste `sql/init.sql`, run it. It also turns on row level security,
   so the public REST API and anon key cannot read or write these tables.
3. **Connect** → copy the **Session pooler** string (port 5432). The direct connection is
   IPv6-only on the free plan, which many hosts cannot reach. The transaction pooler (6543)
   also works: the app disables prepared statements for it.
4. Put your database password into the string and use it as `DATABASE_URL`.

## 2. Load data

From your own machine, pointed at Supabase:

```bash
export DATABASE_URL='postgresql://postgres.<ref>:<password>@<pooler-host>:5432/postgres'
python -m app.ingest docs.jsonl --strategy recursive_256
```

Ingest **one** strategy for the demo. The benchmark's six strategies store the corpus six
times, and the free database is limited to 500 MB (at the time of writing). Check the size
with `SELECT pg_size_pretty(pg_database_size(current_database()));`.

## 3. Redis (Upstash)

Create a free database and use its `rediss://` URL as `REDIS_URL`.

## 4. API host

Any host that runs a Docker image: Render, Railway, Fly.io, or a Hugging Face Space
(Docker SDK). The image honours `PORT`. Models are baked into it, so cold starts do not
download anything. Expect to need around 1 GB of RAM (two models plus PyTorch); measure it
with `docker stats` and avoid 512 MB tiers.

| variable | value |
|---|---|
| `DATABASE_URL` | Supabase pooler string |
| `REDIS_URL` | Upstash `rediss://` URL |
| `ADMIN_TOKEN` | long random string, required on `/ingest` and `DELETE /cache` |
| `RATE_LIMIT_PER_MINUTE` | `30` (per IP, on `/search` and `/ask`) |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | your provider; leave the key empty for retrieval-only |
| `CORS_ORIGINS` | only if a frontend on another origin calls the API |

Set a spending cap with your LLM provider as well. The rate limit is per IP and best effort.

## 5. Check it

```bash
curl https://<api-host>/health
curl -X POST https://<api-host>/search -H 'content-type: application/json' \
  -d '{"query": "does vitamin D reduce fracture risk", "mode": "hybrid_rerank"}'
curl -X DELETE https://<api-host>/cache                                  # expect 401
curl -X DELETE https://<api-host>/cache -H "X-Admin-Token: $ADMIN_TOKEN"  # expect 200
```

`/health` returns the chunk count, so `0` means step 2 pointed at a different database.
