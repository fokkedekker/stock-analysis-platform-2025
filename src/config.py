"""Application configuration using Pydantic settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # FMP API Configuration
    FMP_API_KEY: str
    FMP_BASE_URL: str = "https://financialmodelingprep.com"

    # Database Configuration
    DATABASE_PATH: str = "data/stock_analysis.duckdb"

    # Rate Limiting (700 to stay safely under FMP's 750 limit)
    RATE_LIMIT_PER_MINUTE: int = 700

    # Worker pool for parallel fetching
    NUM_WORKERS: int = 10

    # Retry Configuration
    MAX_RETRIES: int = 5
    RETRY_MIN_WAIT: int = 2
    RETRY_MAX_WAIT: int = 60

    # Checkpoint Configuration
    CHECKPOINT_PATH: str = "data/fetch_checkpoint.json"

    # Logging
    LOG_LEVEL: str = "INFO"

    @property
    def database_path(self) -> Path:
        """Get database path as Path object, creating parent dirs if needed."""
        path = Path(self.DATABASE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
