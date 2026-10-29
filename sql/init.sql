CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id        TEXT PRIMARY KEY,
    title     TEXT,
    source    TEXT
);

-- one row per chunk; `strategy` lets several chunkings of the same corpus
-- live side by side so they can be benchmarked against each other
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy  TEXT NOT NULL,
    idx       INT  NOT NULL,
    text      TEXT NOT NULL,
    tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Semantic cache and rate-limit counters, used when REDIS_URL is not set (serverless deploys).
CREATE TABLE IF NOT EXISTS semantic_cache (
    id         BIGSERIAL PRIMARY KEY,
    scope      TEXT NOT NULL,          -- "<strategy>|<mode>": entries never cross retrieval configs
    query      TEXT NOT NULL,
    embedding  VECTOR(384) NOT NULL,
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS semantic_cache_scope_idx ON semantic_cache (scope, created_at DESC);

CREATE TABLE IF NOT EXISTS rate_limits (
    key    TEXT   NOT NULL,
    bucket BIGINT NOT NULL,            -- unix time // window
    n      INT    NOT NULL DEFAULT 0,
    PRIMARY KEY (key, bucket)
);

-- Supabase serves every table in the public schema over its REST API with the anon
-- key. With RLS enabled and no policies, those roles can read and write nothing. The
-- backend connects as the database owner, which is unaffected, and so is plain Postgres.
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE semantic_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limits    ENABLE ROW LEVEL SECURITY;
