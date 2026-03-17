"""FastAPI route definitions for the prediction service."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EmployeeInput,
    HealthResponse,
    ModelInfoResponse,
    NineBoxRequest,
    NineBoxResponse,
    PredictionResponse,
    ShapDriver,
)
from src.config import get_settings
from src.models.nine_box import NineBoxClassifier
from src.models.recommender import DevelopmentRecommender

router = APIRouter()

_nine_box = NineBoxClassifier()
_recommender = DevelopmentRecommender()


def _employee_to_dataframe(employee: EmployeeInput) -> pd.DataFrame:
    """Convert an EmployeeInput to a single-row DataFrame matching model features."""
    from src.models.predictor import PerformancePredictor

    data: dict[str, Any] = {
        "avg_past_performance": employee.avg_past_performance,
        "performance_trend": employee.performance_trend,
        "total_training_hours": employee.total_training_hours,
        "training_completion_rate": employee.training_completion_rate,
        "avg_training_score": employee.avg_training_score,
        "tenure_years": employee.tenure_years,
        "job_level": employee.job_level,
        "department_encoded": hash(employee.department) % 20,
        "overtime_avg": employee.overtime_hours,
        "goals_met_avg": employee.goals_met_avg,
        "leadership_score": employee.leadership_score,
        "communication_score": employee.communication_score,
        "innovation_score": employee.innovation_score,
    }
    return pd.DataFrame([data], columns=PerformancePredictor.FEATURE_COLUMNS)


def _build_prediction_response(
    employee: EmployeeInput,
    predicted_score: float,
    shap_drivers: list[dict[str, Any]],
) -> PredictionResponse:
    """Assemble a full prediction response with 9-box and recommendations."""
    margin = max(0.15, abs(predicted_score) * 0.08)
    confidence_interval = (
        round(predicted_score - margin, 4),
        round(predicted_score + margin, 4),
    )

    nine_box_result = _nine_box.classify(predicted_score, employee.potential_score)

    profile: dict[str, Any] = {
        "box_label": nine_box_result["box_label"],
        "leadership_score": employee.leadership_score,
        "communication_score": employee.communication_score,
        "innovation_score": employee.innovation_score,
        "training_completion_rate": employee.training_completion_rate,
        "total_training_hours": employee.total_training_hours,
        "performance_trend": employee.performance_trend,
        "goals_met_avg": employee.goals_met_avg,
    }
    recommendations = _recommender.recommend(profile)[:5]

    top_drivers = [
        ShapDriver(
            feature=d["feature"],
            value=d["value"],
            impact=d["shap_value"],
            direction=d["direction"],
        )
        for d in shap_drivers
    ]

    return PredictionResponse(
        predicted_score=round(predicted_score, 4),
        confidence_interval=confidence_interval,
        nine_box_label=nine_box_result["box_label"],
        top_drivers=top_drivers,
        recommendations=recommendations,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(request: Request) -> HealthResponse:
    """Return service health status."""
    settings = get_settings()
    model_loaded: bool = getattr(request.app.state, "model", None) is not None
    start_time: float = getattr(request.app.state, "start_time", time.time())

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.version,
        model_loaded=model_loaded,
        uptime_seconds=round(time.time() - start_time, 2),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(employee: EmployeeInput, request: Request) -> PredictionResponse:
    """Generate a performance prediction for a single employee."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is in degraded mode.",
        )

    X = _employee_to_dataframe(employee)
    prediction = float(model.predict(X)[0])

    shap_drivers: list[dict[str, Any]] = []
    shap_analyzer = getattr(request.app.state, "shap_analyzer", None)
    if shap_analyzer is not None:
        try:
            shap_drivers = shap_analyzer.get_top_drivers(X, index=0, top_n=10)
        except Exception:
            logger.warning("SHAP analysis failed; returning prediction without drivers")

    return _build_prediction_response(employee, prediction, shap_drivers)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Generate performance predictions for a batch of employees."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is in degraded mode.",
        )

    frames = [_employee_to_dataframe(emp) for emp in payload.employees]
    X = pd.concat(frames, ignore_index=True)
    raw_predictions: np.ndarray = model.predict(X)

    shap_analyzer = getattr(request.app.state, "shap_analyzer", None)

    results: list[PredictionResponse] = []
    for idx, employee in enumerate(payload.employees):
        shap_drivers: list[dict[str, Any]] = []
        if shap_analyzer is not None:
            try:
                shap_drivers = shap_analyzer.get_top_drivers(X, index=idx, top_n=10)
            except Exception:
                logger.warning("SHAP analysis failed for employee index {}", idx)

        results.append(
            _build_prediction_response(
                employee, float(raw_predictions[idx]), shap_drivers
            )
        )

    return BatchPredictionResponse(predictions=results)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is in degraded mode.",
        )

    from src.models.predictor import PerformancePredictor

    model_type: str = getattr(request.app.state, "model_type", "unknown")
    metrics: dict[str, float] = getattr(request.app.state, "model_metrics", {})
    training_date: str | None = getattr(request.app.state, "training_date", None)

    return ModelInfoResponse(
        model_type=model_type,
        features=PerformancePredictor.FEATURE_COLUMNS,
        metrics=metrics,
        training_date=training_date,
    )


@router.post("/nine-box", response_model=NineBoxResponse, tags=["analytics"])
async def classify_nine_box(payload: NineBoxRequest) -> NineBoxResponse:
    """Classify an employee into the 9-Box Grid."""
    result = _nine_box.classify(payload.performance_score, payload.potential_score)
    return NineBoxResponse(**result)
