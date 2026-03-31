"""
config/settings.py
─────────────────
Centralised, validated configuration loaded from .env.
All modules import from here — never from os.environ directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"

    # ── Vector store — pgvector (PostgreSQL) ────────────
    postgres_url: str = "postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb"

    # ── Document paths ──────────────────────────────────
    raw_docs_dir: str = "./data/raw"
    processed_docs_dir: str = "./data/processed"

    # ── Chunking ────────────────────────────────────────
    chunk_size: int = Field(default=600, ge=100, le=2000)
    chunk_overlap: int = Field(default=75, ge=0, le=500)

    # ── Retrieval ───────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity_score: float = Field(default=0.30, ge=0.0, le=1.0)

    # ── Confluence (optional) ────────────────────────────
    confluence_url: str = ""
    confluence_username: str = ""
    confluence_api_token: str = ""
    confluence_space_key: str = "ENG"

    # ── Logging ─────────────────────────────────────────
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()
