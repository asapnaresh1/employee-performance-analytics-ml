"""Tests for src.explainability.shap_analyzer.ShapAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap", reason="shap not installed")

from src.explainability.shap_analyzer import ShapAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def shap_fixture():
    """Train a small RandomForest and create a ShapAnalyzer + sample X."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "feat_a": rng.uniform(0, 1, n),
        "feat_b": rng.uniform(0, 1, n),
        "feat_c": rng.uniform(0, 1, n),
    })
    y = X["feat_a"] * 2 + X["feat_b"] + rng.normal(0, 0.1, n)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    analyzer = ShapAnalyzer(model, model_type="tree")
    return analyzer, X


class TestShapAnalyzer:

    def test_analyze_global_keys(self, shap_fixture) -> None:
        """analyze_global returns feature_importance, shap_values, expected_value."""
        analyzer, X = shap_fixture
        result = analyzer.analyze_global(X)
        assert "feature_importance" in result
        assert "shap_values" in result
        assert "expected_value" in result
        assert isinstance(result["feature_importance"], dict)
        assert len(result["feature_importance"]) == X.shape[1]

    def test_analyze_local_keys(self, shap_fixture) -> None:
        """analyze_local returns feature_contributions, base_value, predicted_value."""
        analyzer, X = shap_fixture
        result = analyzer.analyze_local(X, index=0)
        assert "feature_contributions" in result
        assert "base_value" in result
        assert "predicted_value" in result
        assert isinstance(result["feature_contributions"], dict)

    def test_get_top_drivers_count(self, shap_fixture) -> None:
        """get_top_drivers returns at most top_n entries."""
        analyzer, X = shap_fixture
        drivers = analyzer.get_top_drivers(X, index=0, top_n=2)
        assert len(drivers) == 2

    def test_get_top_drivers_has_direction(self, shap_fixture) -> None:
        """Each driver dict contains a 'direction' key with valid values."""
        analyzer, X = shap_fixture
        drivers = analyzer.get_top_drivers(X, index=0, top_n=3)
        for d in drivers:
            assert "direction" in d
            assert d["direction"] in ("positive", "negative")
            assert "feature" in d
            assert "value" in d
            assert "shap_value" in d
