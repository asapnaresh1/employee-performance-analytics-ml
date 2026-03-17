"""Tests for src.models.recommender.DevelopmentRecommender."""

from __future__ import annotations

import pytest

from src.models.recommender import DevelopmentRecommender


@pytest.fixture
def recommender() -> DevelopmentRecommender:
    return DevelopmentRecommender()


class TestDevelopmentRecommender:

    def test_recommend_returns_list(
        self, recommender: DevelopmentRecommender
    ) -> None:
        """recommend() returns a list."""
        profile = {"box_label": "Core Player", "leadership_score": 3.0}
        result = recommender.recommend(profile)
        assert isinstance(result, list)

    def test_recommendation_has_required_keys(
        self, recommender: DevelopmentRecommender
    ) -> None:
        """Each recommendation dict contains area, priority, action, expected_impact."""
        profile = {"box_label": "Under Performer", "leadership_score": 2.0}
        result = recommender.recommend(profile)
        assert len(result) > 0
        required = {"area", "priority", "action", "expected_impact"}
        for rec in result:
            assert required.issubset(set(rec.keys()))

    def test_high_performer_gets_leadership_reco(
        self, recommender: DevelopmentRecommender
    ) -> None:
        """A Star employee receives a retention/growth recommendation."""
        profile = {
            "box_label": "Star",
            "leadership_score": 4.5,
            "communication_score": 4.5,
            "innovation_score": 4.5,
            "training_completion_rate": 0.95,
            "total_training_hours": 100,
            "performance_trend": 0.2,
            "goals_met_avg": 0.9,
        }
        result = recommender.recommend(profile)
        areas = [r["area"] for r in result]
        assert "Retention & Growth" in areas

    def test_low_performer_gets_improvement_reco(
        self, recommender: DevelopmentRecommender
    ) -> None:
        """An Under Performer receives a performance improvement recommendation."""
        profile = {
            "box_label": "Under Performer",
            "leadership_score": 1.5,
            "communication_score": 1.5,
            "innovation_score": 1.5,
            "training_completion_rate": 0.3,
            "total_training_hours": 10,
            "performance_trend": -0.3,
            "goals_met_avg": 0.2,
        }
        result = recommender.recommend(profile)
        areas = [r["area"] for r in result]
        assert "Performance Improvement" in areas
