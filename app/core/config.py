from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_ttl_seconds: int = 3600

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"

    chunk_size: int = 800
    chunk_overlap: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
