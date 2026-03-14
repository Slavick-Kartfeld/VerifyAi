from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Database — Railway injects DATABASE_URL automatically
    # Fallback to SQLite for simple deploys without external DB
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./verifyai.db"
    )

    # Redis (optional for Railway)
    REDIS_URL: str = "redis://redis:6379/0"

    # AWS
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "eu-west-1"
    S3_BUCKET_NAME: str = "verifyai-media"

    # AI
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # HITL
    HITL_CONFIDENCE_THRESHOLD: float = 0.75

    # Security
    SECRET_KEY: str = "change-me-in-production"
    API_KEY_SALT: str = "change-me-too"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # Railway gives postgres:// but asyncpg needs postgresql+asyncpg://
    url = settings.DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    settings.DATABASE_URL = url
    return settings
