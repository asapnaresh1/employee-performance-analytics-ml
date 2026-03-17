"""Tests for src.models.clustering.EmployeeProfileClusterer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.clustering import EmployeeProfileClusterer


@pytest.fixture
def cluster_df() -> pd.DataFrame:
    """Minimal DataFrame with a subset of expected feature columns."""
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame({
        "performance_score": rng.uniform(1, 5, n),
        "potential_score": rng.uniform(1, 5, n),
        "leadership_score": rng.uniform(1, 5, n),
        "communication_score": rng.uniform(1, 5, n),
        "innovation_score": rng.uniform(1, 5, n),
        "total_training_hours": rng.uniform(10, 200, n),
        "training_completion_rate": rng.uniform(0.3, 1.0, n),
        "avg_training_score": rng.uniform(50, 100, n),
        "tenure_years": rng.uniform(0.5, 15, n),
        "overtime_avg": rng.uniform(0, 40, n),
        "goals_met_avg": rng.uniform(0.2, 1.0, n),
    })


class TestEmployeeProfileClusterer:

    def test_fit_predict_returns_labels(self, cluster_df: pd.DataFrame) -> None:
        """fit_predict returns a DataFrame with a 'cluster' column."""
        clusterer = EmployeeProfileClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(cluster_df)
        assert "cluster" in result.columns
        assert len(result) == len(cluster_df)

    def test_cluster_count_matches(self, cluster_df: pd.DataFrame) -> None:
        """Number of unique clusters equals n_clusters."""
        k = 4
        clusterer = EmployeeProfileClusterer(n_clusters=k, random_state=42)
        result = clusterer.fit_predict(cluster_df)
        assert result["cluster"].nunique() == k

    def test_get_cluster_profiles_shape(self, cluster_df: pd.DataFrame) -> None:
        """get_cluster_profiles returns one row per cluster."""
        k = 3
        clusterer = EmployeeProfileClusterer(n_clusters=k, random_state=42)
        result = clusterer.fit_predict(cluster_df)
        profiles = clusterer.get_cluster_profiles(result)
        assert len(profiles) == k
        assert "cluster" in profiles.columns

    def test_get_optimal_k_returns_scores(self, cluster_df: pd.DataFrame) -> None:
        """get_optimal_k returns silhouette_scores and recommended_k."""
        clusterer = EmployeeProfileClusterer(random_state=42)
        result = clusterer.get_optimal_k(cluster_df, k_range=range(2, 6))
        assert "silhouette_scores" in result
        assert "recommended_k" in result
        assert isinstance(result["silhouette_scores"], dict)
        assert result["recommended_k"] in range(2, 6)
