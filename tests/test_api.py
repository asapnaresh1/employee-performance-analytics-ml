"""Tests for the FastAPI application (src.api)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app (no model loaded -> degraded mode)."""
    app = create_app()
    return TestClient(app)


class TestAPI:

    def test_health_endpoint(self, client: TestClient) -> None:
        """GET /api/v1/health returns 200 with expected fields."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_nine_box_endpoint(self, client: TestClient) -> None:
        """POST /api/v1/nine-box classifies correctly."""
        payload = {"performance_score": 4.5, "potential_score": 4.5}
        response = client.post("/api/v1/nine-box", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["box_label"] == "Star"
        assert data["box_number"] == 1
        assert "recommendation" in data

    def test_predict_returns_503_without_model(self, client: TestClient) -> None:
        """POST /api/v1/predict returns 503 when no model is loaded (degraded mode)."""
        payload = {
            "age": 30,
            "department": "Engineering",
            "job_level": 3,
            "tenure_years": 5.0,
            "avg_past_performance": 3.5,
            "performance_trend": 0.1,
            "total_training_hours": 50,
            "training_completion_rate": 0.8,
            "avg_training_score": 75.0,
            "overtime_hours": 10,
            "goals_met_avg": 0.7,
            "leadership_score": 3.5,
            "communication_score": 3.5,
            "innovation_score": 3.5,
            "satisfaction_score": 3.5,
            "potential_score": 3.5,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 503
