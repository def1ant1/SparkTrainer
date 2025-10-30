"""
Configuration management using Pydantic for environment variable validation.

This module provides type-safe configuration loading from environment variables
with validation and default values.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable validation.

    All settings are loaded from environment variables with validation.
    Missing required variables will raise a ValidationError.
    """

    # Database Configuration
    database_url: str = Field(
        default="postgresql://sparktrainer:sparktrainer@localhost:5432/sparktrainer",
        description="PostgreSQL database connection URL",
        validation_alias="DATABASE_URL",
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
        validation_alias="REDIS_URL",
    )

    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
        validation_alias="CELERY_BROKER_URL",
    )

    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
        validation_alias="CELERY_RESULT_BACKEND",
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLflow tracking server URI",
        validation_alias="MLFLOW_TRACKING_URI",
    )

    mlflow_artifact_root: str = Field(
        default="./mlruns",
        description="MLflow artifact storage root directory",
        validation_alias="MLFLOW_ARTIFACT_ROOT",
    )

    # Flask Configuration
    flask_env: str = Field(
        default="production",
        description="Flask environment (development/production)",
        validation_alias="FLASK_ENV",
    )

    flask_debug: bool = Field(
        default=False,
        description="Enable Flask debug mode",
        validation_alias="FLASK_DEBUG",
    )

    secret_key: str = Field(
        default="",
        description="Flask secret key for sessions",
        validation_alias="SECRET_KEY",
    )

    # Training Configuration
    dgx_trainer_base_dir: str = Field(
        default="/app",
        description="Base directory for training operations",
        validation_alias="DGX_TRAINER_BASE_DIR",
    )

    cuda_visible_devices: str = Field(
        default="0",
        description="Comma-separated list of CUDA device IDs",
        validation_alias="CUDA_VISIBLE_DEVICES",
    )

    # Storage Configuration
    storage_backend: str = Field(
        default="local",
        description="Storage backend (local/s3/gcs)",
        validation_alias="STORAGE_BACKEND",
    )

    local_storage_path: str = Field(
        default="./storage",
        description="Local storage directory path",
        validation_alias="LOCAL_STORAGE_PATH",
    )

    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name for artifact storage",
        validation_alias="S3_BUCKET",
    )

    s3_region: Optional[str] = Field(
        default=None,
        description="S3 region",
        validation_alias="S3_REGION",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
        validation_alias="API_HOST",
    )

    api_port: int = Field(
        default=5000,
        description="API server port",
        validation_alias="API_PORT",
    )

    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins",
        validation_alias="CORS_ORIGINS",
    )

    # Authentication (optional)
    enable_auth: bool = Field(
        default=False,
        description="Enable authentication",
        validation_alias="ENABLE_AUTH",
    )

    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="JWT secret key for token signing",
        validation_alias="JWT_SECRET_KEY",
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
        validation_alias="JWT_ALGORITHM",
    )

    # Testing
    testing: bool = Field(
        default=False,
        description="Enable testing mode",
        validation_alias="TESTING",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate that secret_key is set in production."""
        flask_env = info.data.get("flask_env", "production")
        if flask_env == "production" and not v:
            raise ValueError(
                "SECRET_KEY must be set in production environment. "
                "Generate a secure random key using: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        return v

    @field_validator("storage_backend")
    @classmethod
    def validate_storage_backend(cls, v: str) -> str:
        """Validate storage backend value."""
        allowed_backends = {"local", "s3", "gcs"}
        if v not in allowed_backends:
            raise ValueError(f"storage_backend must be one of {allowed_backends}, got: {v}")
        return v

    @field_validator("s3_bucket")
    @classmethod
    def validate_s3_config(cls, v: Optional[str], info) -> Optional[str]:
        """Validate S3 configuration when S3 backend is used."""
        storage_backend = info.data.get("storage_backend")
        if storage_backend == "s3" and not v:
            raise ValueError("S3_BUCKET must be set when using S3 storage backend")
        return v

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: Optional[str], info) -> Optional[str]:
        """Validate JWT secret key when authentication is enabled."""
        enable_auth = info.data.get("enable_auth", False)
        if enable_auth and not v:
            raise ValueError(
                "JWT_SECRET_KEY must be set when authentication is enabled. "
                "Generate a secure random key using: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        return v

    @field_validator("cuda_visible_devices")
    @classmethod
    def validate_cuda_devices(cls, v: str) -> str:
        """Validate CUDA device specification."""
        if v and v != "":
            # Validate format: comma-separated integers
            try:
                devices = [int(d.strip()) for d in v.split(",")]
                if any(d < 0 for d in devices):
                    raise ValueError("CUDA device IDs must be non-negative integers")
            except ValueError as e:
                raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES format: {e}")
        return v

    def get_cors_origins_list(self) -> list:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings instance.

    Returns:
        Settings: Validated application settings
    """
    return settings


def validate_settings() -> None:
    """
    Validate all settings and print configuration summary.

    Raises:
        ValidationError: If any settings are invalid
    """
    try:
        config = get_settings()
        print("=" * 60)
        print("SparkTrainer Configuration")
        print("=" * 60)
        print(f"Environment: {config.flask_env}")
        print(f"Debug Mode: {config.flask_debug}")
        print(f"Database: {config.database_url}")
        print(f"Redis: {config.redis_url}")
        print(f"MLflow: {config.mlflow_tracking_uri}")
        print(f"Storage Backend: {config.storage_backend}")
        print(f"API: {config.api_host}:{config.api_port}")
        print(f"Authentication: {'Enabled' if config.enable_auth else 'Disabled'}")
        print("=" * 60)
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # Validate configuration when run as a script
    validate_settings()
