"""Application settings.

All configuration is environment-driven (12-factor). See ``.env.example`` for
the full list of supported keys and their defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded once per process from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ── General ───────────────────────────────────────────────────────────────
    app_name: str = "AI Meeting Assistant"
    app_env: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=list)

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/meetings.db"
    db_echo: bool = False
    auto_create_tables: bool = True  # disabled automatically in production

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    password_hash_iterations: int = 310_000

    # ── Uploads / storage ─────────────────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    max_upload_bytes: int = 300 * 1024 * 1024  # 300 MB
    allowed_audio_extensions: set[str] = {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
    }
    upload_chunk_bytes: int = 1024 * 1024
    orphan_upload_ttl_hours: int = 24

    # ── Transcription / diarization ───────────────────────────────────────────
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto | cpu | cuda
    transcribe_timeout_seconds: int = 1800
    diarization_enabled: bool = True
    hf_token: str | None = None
    pyannote_pipeline: str = "pyannote/speaker-diarization-3.1"
    diarization_max_speakers: int | None = None

    # ── Translation ───────────────────────────────────────────────────────────
    translate_enabled: bool = True
    translation_source_language: str = "hi"
    translation_target_language: str = "en"
    translation_batch_size: int = 20

    # ── LLM ───────────────────────────────────────────────────────────────────
    mistral_api_key: str | None = None
    mistral_model: str = "mistral-small-latest"
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.2

    # ── RAG ───────────────────────────────────────────────────────────────────
    vector_store: Literal["memory", "chroma"] = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    embedding_backend: Literal["hashing", "sentence-transformers"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    rag_top_k: int = 6
    rag_min_score: float = 0.15  # cosine-similarity floor for retrieval
    rag_chunk_max_chars: int = 1200
    rag_chunk_overlap_chars: int = 200
    chat_history_max_messages: int = 8

    # ── Jobs / workers ────────────────────────────────────────────────────────
    job_workers: int = 2
    job_poll_interval_seconds: float = 2.0
    job_max_attempts: int = 3
    job_retry_backoff_seconds: int = 30
    job_stale_after_seconds: int = 3600
    pipeline_stage_timeout_seconds: int = 3600

    # ── Validation ────────────────────────────────────────────────────────────

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def effective_auto_create_tables(self) -> bool:
        """In production, schema changes must go through Alembic migrations."""
        return self.auto_create_tables and not self.is_production


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (tests reset the cache instead of mutating)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings — used by the test-suite."""
    get_settings.cache_clear()
