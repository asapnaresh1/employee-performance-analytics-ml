"""Tests for src.models.predictor.PerformancePredictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.predictor import PerformancePredictor


@pytest.fixture
def simple_features() -> tuple[pd.DataFrame, pd.Series]:
    """Return a minimal (X, y) pair suitable for the predictor."""
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({
        "avg_past_performance": rng.uniform(1, 5, n),
        "performance_trend": rng.normal(0, 0.2, n),
        "total_training_hours": rng.uniform(10, 200, n),
        "training_completion_rate": rng.uniform(0.3, 1.0, n),
        "avg_training_score": rng.uniform(50, 100, n),
        "tenure_years": rng.uniform(0.5, 15, n),
        "job_level": rng.integers(1, 7, n),
        "department_encoded": rng.integers(0, 8, n),
        "overtime_avg": rng.uniform(0, 40, n),
        "goals_met_avg": rng.uniform(0.2, 1.0, n),
        "leadership_score": rng.uniform(1, 5, n),
        "communication_score": rng.uniform(1, 5, n),
        "innovation_score": rng.uniform(1, 5, n),
    })
    y = pd.Series(rng.uniform(1, 5, n), name="performance_score")
    return X, y


class TestPerformancePredictor:

    def test_prepare_features_columns(
        self,
        sample_employees: pd.DataFrame,
        sample_evaluations: pd.DataFrame,
        sample_training: pd.DataFrame,
    ) -> None:
        """prepare_features produces a DataFrame with employee_id and numeric features."""
        evaluations = sample_evaluations.copy()
        training = sample_training.copy()
        employees = sample_employees.copy()

        # Adapt column names
        if "hours" in training.columns:
            training = training.rename(columns={"hours": "training_hours"})
        if "status" in training.columns:
            training["completion_status"] = (training["status"] == "Completed").astype(float)
        if "goals_met_pct" in evaluations.columns:
            evaluations["goals_met"] = evaluations["goals_met_pct"] / 100.0

        features = PerformancePredictor.prepare_features(evaluations, training, employees)
        assert "employee_id" in features.columns
        assert len(features) > 0

    def test_fit_returns_metrics(
        self, simple_features: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """fit() returns a dict with r2, mae, rmse."""
        X, y = simple_features
        predictor = PerformancePredictor(model_type="random_forest", random_state=42)
        metrics = predictor.fit(X, y)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["r2"] >= 0  # in-sample R2 should be non-negative

    def test_predict_shape(
        self, simple_features: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """predict() returns array with same length as input."""
        X, y = simple_features
        predictor = PerformancePredictor(model_type="random_forest", random_state=42)
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_cross_validate_folds(
        self, simple_features: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """cross_validate returns fold_r2_scores with the correct number of folds."""
        X, y = simple_features
        predictor = PerformancePredictor(model_type="random_forest", random_state=42)
        cv_result = predictor.cross_validate(X, y, cv=3)
        assert "fold_r2_scores" in cv_result
        assert len(cv_result["fold_r2_scores"]) == 3
        assert "mean_r2" in cv_result
        assert "std_r2" in cv_result

    def test_save_and_load(
        self,
        simple_features: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Model can be saved and reloaded, producing the same predictions."""
        X, y = simple_features
        predictor = PerformancePredictor(model_type="random_forest", random_state=42)
        predictor.fit(X, y)
        original_preds = predictor.predict(X)

        model_path = tmp_path / "model.joblib"
        predictor.save(model_path)

        loaded = PerformancePredictor(model_type="random_forest", random_state=42)
        loaded.load(model_path)
        loaded_preds = loaded.predict(X)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    @pytest.mark.parametrize(
        "model_type",
        ["random_forest", "gradient_boosting"],
    )
    def test_model_types(
        self,
        model_type: str,
        simple_features: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Each supported model type can fit and predict without error."""
        X, y = simple_features
        predictor = PerformancePredictor(model_type=model_type, random_state=42)
        metrics = predictor.fit(X, y)
        assert isinstance(metrics, dict)
        preds = predictor.predict(X)
        assert len(preds) == len(X)
