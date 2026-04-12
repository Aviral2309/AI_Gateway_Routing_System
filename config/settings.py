from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini
    gemini_api_key: Optional[str] = None
    gemini_primary_model: str = "gemini-2.5-flash"
    gemini_lite_model: str = "gemini-2.5-flash-lite-preview-06-17"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_router"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # Routing
    default_budget_usd: float = 5.00
    max_retries: int = 3
    enable_evaluation: bool = False  # off by default to save tokens

    # Agents
    enable_agents: bool = True
    agent_max_steps: int = 5

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # App
    log_level: str = "INFO"
    app_version: str = "2.0.0"


settings = Settings()

# Model cost per 1K tokens (USD) — Gemini 2.5 pricing
MODEL_COSTS = {
    "gemini-2.5-flash":                    {"input": 0.00015,  "output": 0.0006},
    "gemini-2.5-flash-lite-preview-06-17": {"input": 0.000075, "output": 0.0003},
    "ollama-local":                         {"input": 0.0,      "output": 0.0},
}
