"""Tests for src.config settings."""

from __future__ import annotations

from src.config import APISettings, AppSettings, DataSettings, ModelSettings, get_settings


class TestConfig:

    def test_data_settings_defaults(self) -> None:
        """DataSettings has correct default values."""
        ds = DataSettings()
        assert ds.n_employees == 1500
        assert ds.random_seed == 42
        assert ds.train_ratio == 0.7
        assert len(ds.evaluation_periods) == 8

    def test_model_settings_defaults(self) -> None:
        """ModelSettings has correct default values."""
        ms = ModelSettings()
        assert ms.target_column == "performance_score"
        assert ms.n_estimators == 300
        assert ms.max_depth == 8
        assert ms.learning_rate == 0.05
        assert ms.cv_folds == 5

    def test_api_settings_defaults(self) -> None:
        """APISettings has correct default values."""
        api = APISettings()
        assert api.host == "0.0.0.0"
        assert api.port == 8000
        assert api.workers == 1
        assert api.cors_origins == ["*"]

    def test_get_settings_singleton(self) -> None:
        """get_settings returns the same cached instance on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        assert isinstance(s1, AppSettings)
        assert s1.app_name == "Employee Performance Analytics ML"
        assert s1.version == "1.0.0"
