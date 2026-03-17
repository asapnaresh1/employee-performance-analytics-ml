"""Shared fixtures for the employee-performance-analytics-ml test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import PerformanceDataGenerator

# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive — generated once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def all_tables() -> dict[str, pd.DataFrame]:
    """Generate all four synthetic tables with a small employee count."""
    gen = PerformanceDataGenerator(n_employees=50, random_seed=99)
    return gen.generate_all()


@pytest.fixture(scope="session")
def sample_employees(all_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Small employees DataFrame (~50 rows)."""
    return all_tables["employees"]


@pytest.fixture(scope="session")
def sample_evaluations(all_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluations for the sample employees (multiple periods)."""
    return all_tables["evaluations"]


@pytest.fixture(scope="session")
def sample_training(all_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Training records for the sample employees."""
    return all_tables["training_records"]


@pytest.fixture(scope="session")
def sample_promotions(all_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Promotions for the sample employees."""
    return all_tables["promotions"]


# ---------------------------------------------------------------------------
# Fitted predictor (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_predictor(
    sample_employees: pd.DataFrame,
    sample_evaluations: pd.DataFrame,
    sample_training: pd.DataFrame,
):
    """Return a PerformancePredictor fitted on sample data.

    Uses ``random_forest`` to avoid a hard dependency on xgboost in CI.
    """
    from src.models.predictor import PerformancePredictor

    # Build a minimal feature matrix that the predictor can consume.
    # We need columns it expects: employee_id + engineered features.
    evaluations = sample_evaluations.copy()
    training = sample_training.copy()
    employees = sample_employees.copy()

    # Adapt column names to what prepare_features expects
    if "hours" in training.columns and "training_hours" not in training.columns:
        training = training.rename(columns={"hours": "training_hours"})
    if "status" in training.columns and "completion_status" not in training.columns:
        training["completion_status"] = (training["status"] == "Completed").astype(float)
    if "goals_met_pct" in evaluations.columns and "goals_met" not in evaluations.columns:
        evaluations["goals_met"] = evaluations["goals_met_pct"] / 100.0

    features = PerformancePredictor.prepare_features(evaluations, training, employees)

    # Target: average performance score per employee
    target = (
        evaluations.groupby("employee_id")["performance_score"]
        .mean()
        .reset_index(name="target")
    )
    merged = features.merge(target, on="employee_id", how="inner")

    y = merged["target"]
    X = merged.drop(columns=["employee_id", "target"], errors="ignore")
    # Keep only numeric columns that exist
    X = X.select_dtypes(include=[np.number])

    predictor = PerformancePredictor(model_type="random_forest", random_state=42)
    predictor.fit(X, y)
    predictor._X_sample = X  # stash for downstream tests
    predictor._y_sample = y
    return predictor
