"""Tests for src.causal.inference.CausalAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

dowhy = pytest.importorskip("dowhy", reason="dowhy not installed")

from src.causal.inference import CausalAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def causal_data() -> pd.DataFrame:
    """Create a small synthetic dataset with known causal structure."""
    rng = np.random.default_rng(42)
    n = 200
    training_hours = rng.uniform(10, 200, n)
    tenure_years = rng.uniform(0.5, 15, n)
    overtime_hours = rng.uniform(0, 40, n)
    manager_rating = rng.uniform(1, 5, n)
    noise = rng.normal(0, 0.3, n)
    performance_score = (
        0.01 * training_hours
        + 0.05 * tenure_years
        - 0.02 * overtime_hours
        + 0.3 * manager_rating
        + noise
    )
    return pd.DataFrame({
        "total_training_hours": training_hours,
        "tenure_years": tenure_years,
        "overtime_hours": overtime_hours,
        "manager_rating": manager_rating,
        "performance_score": performance_score,
    })


@pytest.fixture
def analyzer() -> CausalAnalyzer:
    return CausalAnalyzer(random_seed=42)


class TestCausalAnalyzer:

    def test_estimate_effect_returns_dict(
        self, analyzer: CausalAnalyzer, causal_data: pd.DataFrame
    ) -> None:
        """estimate_effect returns a dictionary."""
        result = analyzer.estimate_effect(
            data=causal_data,
            treatment="total_training_hours",
            outcome="performance_score",
        )
        assert isinstance(result, dict)

    def test_estimate_effect_has_ate(
        self, analyzer: CausalAnalyzer, causal_data: pd.DataFrame
    ) -> None:
        """estimate_effect result contains ate, ci_lower, ci_upper."""
        result = analyzer.estimate_effect(
            data=causal_data,
            treatment="total_training_hours",
            outcome="performance_score",
        )
        assert "ate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert isinstance(result["ate"], float)

    def test_run_hypothesis_tests_returns_list(
        self, analyzer: CausalAnalyzer, causal_data: pd.DataFrame
    ) -> None:
        """run_hypothesis_tests returns a list of result dicts."""
        results = analyzer.run_hypothesis_tests(causal_data)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "hypothesis" in r or "status" in r

    def test_summarize_findings_returns_string(
        self, analyzer: CausalAnalyzer, causal_data: pd.DataFrame
    ) -> None:
        """summarize_findings produces a non-empty string."""
        results = analyzer.run_hypothesis_tests(causal_data)
        summary = analyzer.summarize_findings(results)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Causal Analysis Summary" in summary
