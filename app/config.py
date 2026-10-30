from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    # empty means the semantic cache and rate limits live in Postgres (serverless deploys)
    redis_url: str = ""
    db_pool_max: int = 8  # keep small (2-3) on serverless: every instance has its own pool

    # ONNX exports of BAAI/bge-small-en-v1.5 and cross-encoder/ms-marco-MiniLM-L-6-v2 (see
    # scripts/export_models.py). Empty means the models/ folder at the repo root.
    models_dir: str = ""
    embed_dim: int = 384

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # sent as `reasoning_effort` when set ("low" keeps Gemini's thinking models quick).
    # Leave empty for endpoints that reject the field, such as OpenAI's chat models.
    llm_reasoning_effort: str = ""
    # seconds to wait for the next chunk from the model before giving up; must stay below the
    # platform's function limit so the page gets an error message instead of a dead stream
    llm_timeout_seconds: float = 40.0

    # cosine similarity above which a cached answer is reused. 0.96 reuses about 96% of
    # paraphrases and wrongly serves about 6% of distinct questions (bench/cache_bench.py);
    # the old 0.92 default wrongly served 12%.
    cache_threshold: float = 0.96
    cache_ttl_seconds: int = 3600

    default_strategy: str = "recursive_256"

    # weight of the full-text list in rank fusion (the vector list is 1.0). Postgres full-text
    # ranking is a much weaker retriever than the vectors on SciFact, and nDCG@10 rises steadily
    # as its weight falls: 0.627 at 1.0, 0.678 at 0.25, 0.705 at 0.1 (README, Results).
    lexical_weight: float = 0.1

    # HNSW candidate list size per scan batch (pgvector hnsw.ef_search)
    hnsw_ef_search: int = 100

    # public-deployment guards. Empty ADMIN_TOKEN leaves /ingest and DELETE /cache open (local dev).
    admin_token: str = ""
    rate_limit_per_minute: int = 30  # per client IP on /search and /ask; 0 disables
    cors_origins: str = ""  # comma-separated; only needed if a separate frontend calls the API


settings = Settings()
