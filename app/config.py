from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_dim: int = 384

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # cosine similarity above which a cached answer is reused
    cache_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    default_strategy: str = "recursive_256"

    # HNSW candidate list size per scan batch (pgvector hnsw.ef_search)
    hnsw_ef_search: int = 100

    # public-deployment guards. Empty ADMIN_TOKEN leaves /ingest and DELETE /cache open (local dev).
    admin_token: str = ""
    rate_limit_per_minute: int = 30  # per client IP on /search and /ask; 0 disables
    cors_origins: str = ""  # comma-separated; only needed if a separate frontend calls the API


settings = Settings()
