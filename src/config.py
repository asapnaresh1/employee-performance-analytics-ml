"""
Application configuration using Pydantic BaseSettings.

Loads settings from environment variables and .env files with
type validation and sensible defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DataSettings(BaseSettings):
    """Settings related to data generation and storage."""

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    n_employees: int = Field(default=1500, ge=100, le=50_000, description="Number of synthetic employees to generate")
    evaluation_periods: list[str] = Field(
        default=["H1-2021", "H2-2021", "H1-2022", "H2-2022", "H1-2023", "H2-2023", "H1-2024", "H2-2024"],
        description="Semi-annual evaluation periods",
    )
    output_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw", description="Directory for generated CSV files")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.95, description="Train split ratio")
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.4, description="Validation split ratio")
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.4, description="Test split ratio")


class ModelSettings(BaseSettings):
    """Settings for ML model training and evaluation."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    target_column: str = Field(default="performance_score", description="Target variable for prediction")
    n_estimators: int = Field(default=300, ge=10, le=5000, description="Number of estimators for ensemble models")
    max_depth: int | None = Field(default=8, ge=1, le=50, description="Maximum tree depth")
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0, description="Learning rate for gradient boosting")
    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of cross-validation folds")
    early_stopping_rounds: int = Field(default=30, ge=5, le=200, description="Early stopping patience")
    model_dir: Path = Field(default=PROJECT_ROOT / "models", description="Directory to save trained models")
    mlflow_tracking_uri: str = Field(default="mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="employee-performance", description="MLflow experiment name")


class APISettings(BaseSettings):
    """Settings for the FastAPI prediction service."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of Uvicorn workers")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    api_key: str | None = Field(default=None, description="Optional API key for authentication")
    log_level: str = Field(default="info", description="Logging level")


class AppSettings(BaseSettings):
    """Root settings that aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Employee Performance Analytics ML", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")

    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached singleton instance of AppSettings."""
    return AppSettings()
