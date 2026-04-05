from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nexus_router"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # Routing
    default_budget_usd: float = 5.00
    max_retries: int = 2
    enable_evaluation: bool = True

    # Agents
    enable_agents: bool = True
    agent_max_steps: int = 5

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # App
    log_level: str = "INFO"
    app_version: str = "1.0.0"


settings = Settings()

MODEL_COSTS = {
    "gemini-2.5-flash-lite": {"input": 0.000035, "output": 0.000105},  # ultra cheap
    "gemini-2.5-flash":      {"input": 0.00035,  "output": 0.00105},   # 10x flash-lite
    "gemini-1.5-pro":        {"input": 0.0035,   "output": 0.0105},    # high quality
    "ollama-local":          {"input": 0.0,      "output": 0.0},       # local = free
}
