"""
Configuration management for ML Service
Uses pydantic-settings for type-safe environment variables
"""

import warnings
from typing import Optional, List
from pydantic import ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


# Minimum secret key length for production security
MIN_SECRET_KEY_LENGTH = 32

# Default secret key (only allowed in development)
DEFAULT_SECRET_KEY = "dev-secret-key-change-this-in-production"


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
    skip_model_warmup: bool = (
        False  # Set to True for faster dev startup (first request will be slow)
    )

    # Performance Settings
    # FAST_MODE: Disables OWL-ViT multi-food detection for ~5x faster inference
    # Set to True for development/testing when speed > accuracy
    fast_mode: bool = False
    # ENABLE_MULTI_FOOD: Enable OWL-ViT multi-food detection (slow but accurate)
    # Ignored if fast_mode is True
    enable_multi_food: bool = True
    # USE_ONNX: Use ONNX Runtime for 2-3x faster CPU inference
    # Recommended for CPU deployments, ignored when CUDA is available
    use_onnx: bool = False
    # DEVICE: Force specific compute device (auto, cpu, cuda, mps)
    # "auto" will detect available hardware in order: cuda > mps > cpu
    compute_device: str = "auto"

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

    # Sentry Error Tracking (optional)
    sentry_dsn: Optional[str] = None
    sentry_traces_sample_rate: float = 0.1  # 10% of transactions in production

    # Security (for JWT validation and API auth)
    secret_key: str = DEFAULT_SECRET_KEY
    algorithm: str = "HS256"

    @model_validator(mode="after")
    def validate_secret_key_security(self) -> "Settings":
        """
        Validate secret_key meets security requirements.

        Production requirements:
        - Must not use default value
        - Must be at least MIN_SECRET_KEY_LENGTH characters

        Development:
        - Warns if using default value
        """
        is_production = self.environment.lower() == "production"
        is_default = self.secret_key == DEFAULT_SECRET_KEY
        is_too_short = len(self.secret_key) < MIN_SECRET_KEY_LENGTH

        if is_production:
            if is_default:
                raise ValueError(
                    f"SECRET_KEY must be set in production. "
                    f"Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            if is_too_short:
                raise ValueError(
                    f"SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters "
                    f"in production. Current length: {len(self.secret_key)}"
                )
        else:
            # Development/test environment warnings
            if is_default:
                warnings.warn(
                    "Using default SECRET_KEY - this is insecure and should only "
                    "be used in development. Set SECRET_KEY environment variable.",
                    UserWarning,
                    stacklevel=2,
                )
            elif is_too_short:
                warnings.warn(
                    f"SECRET_KEY is shorter than recommended ({MIN_SECRET_KEY_LENGTH} chars). "
                    f"Consider using a longer key for better security.",
                    UserWarning,
                    stacklevel=2,
                )

        return self

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
