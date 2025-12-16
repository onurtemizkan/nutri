"""
Configuration management for ML Service
Uses pydantic-settings for type-safe environment variables
"""

from typing import Optional, List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Nutri ML Service"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Database (PostgreSQL)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nutri_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_max_connections: int = 10

    # Cache TTL (in seconds)
    cache_ttl_features: int = 3600  # 1 hour
    cache_ttl_predictions: int = 86400  # 24 hours
    cache_ttl_models: int = 604800  # 7 days

    # ML Models
    model_storage_path: str = "./app/ml_models"
    model_version: str = "v1.0.0"

    # Feature Engineering
    feature_version: str = "v1.2.3"
    min_data_points_for_ml: int = 30  # Minimum days of data required

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"

    # Security (for future JWT validation)
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to create singleton pattern.
    """
    return Settings()


# Convenience accessor
settings = get_settings()
