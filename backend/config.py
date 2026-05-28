from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    `.env` is used for local development only. Production deployments should
    inject the same variables through their secret/config mechanism.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Teacher Benchmark Arena API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model_name: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL_NAME")
    gemini_timeout_seconds: float = Field(default=75.0, alias="GEMINI_TIMEOUT_SECONDS", gt=0)
    gemini_max_retries: int = Field(default=1, alias="GEMINI_MAX_RETRIES", ge=0, le=5)
    benchmark_judge_delay_seconds: float = Field(default=13.0, alias="BENCHMARK_JUDGE_DELAY_SECONDS", ge=0)

    qwen_base_url: str = Field(default="http://10.240.166.9:8001/v1", alias="QWEN_BASE_URL")
    qwen_model_name: str = Field(default="Qwen/Qwen3-8B", alias="QWEN_MODEL_NAME")
    qwen_timeout_seconds: float = Field(default=90.0, alias="QWEN_TIMEOUT_SECONDS", gt=0)
    qwen_max_tokens: int = Field(default=500, alias="QWEN_MAX_TOKENS", ge=1, le=4096)

    gemma_base_url: str = Field(default="http://127.0.0.1:11435", alias="GEMMA_BASE_URL")
    gemma_model_name: str = Field(default="gemma4:e4b-it-q8_0", alias="GEMMA_MODEL_NAME")
    gemma_timeout_seconds: float = Field(default=120.0, alias="GEMMA_TIMEOUT_SECONDS", gt=0)
    gemma_max_tokens: int = Field(default=500, alias="GEMMA_MAX_TOKENS", ge=1, le=4096)

    llama_base_url: str = Field(default="http://127.0.0.1:11435", alias="LLAMA_BASE_URL")
    llama_model_name: str = Field(default="llama3.1:8b", alias="LLAMA_MODEL_NAME")
    llama_timeout_seconds: float = Field(default=120.0, alias="LLAMA_TIMEOUT_SECONDS", gt=0)
    llama_max_tokens: int = Field(default=500, alias="LLAMA_MAX_TOKENS", ge=1, le=4096)

    model_autostart_enabled: bool = Field(default=True, alias="MODEL_AUTOSTART_ENABLED")
    model_startup_timeout_seconds: float = Field(default=90.0, alias="MODEL_STARTUP_TIMEOUT_SECONDS", gt=0)
    qwen_start_command: str = Field(default="", alias="QWEN_START_COMMAND")
    gemma_start_command: str = Field(default="", alias="GEMMA_START_COMMAND")
    llama_start_command: str = Field(default="", alias="LLAMA_START_COMMAND")

    database_url: str = Field(
        default="postgresql+asyncpg://llm_compare:llm_compare@127.0.0.1:5432/llm_compare",
        alias="DATABASE_URL",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")
    database_pool_size: int = Field(default=5, alias="DATABASE_POOL_SIZE", ge=1)
    database_max_overflow: int = Field(default=10, alias="DATABASE_MAX_OVERFLOW", ge=0)

    elo_starting_score: float = Field(default=1000.0, alias="ELO_STARTING_SCORE")
    elo_k_factor: float = Field(default=32.0, alias="ELO_K_FACTOR", gt=0)


@lru_cache
def get_settings() -> Settings:
    """Cache settings so all modules share one parsed configuration object."""

    return Settings()
