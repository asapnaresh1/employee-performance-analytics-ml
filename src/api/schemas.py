"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Input features required for a single employee performance prediction."""

    age: int = Field(..., ge=18, le=70, description="Employee age in years")
    department: str = Field(..., description="Department name (e.g. Engineering, Sales)")
    job_level: int = Field(..., ge=1, le=10, description="Job level within the organisation")
    tenure_years: float = Field(..., ge=0, description="Years of tenure at the company")
    avg_past_performance: float = Field(
        ..., ge=0, le=5, description="Average historical performance score"
    )
    performance_trend: float = Field(
        0.0, description="Slope of performance over evaluation periods"
    )
    total_training_hours: float = Field(0, ge=0, description="Cumulative training hours")
    training_completion_rate: float = Field(
        1.0, ge=0, le=1, description="Fraction of assigned training completed"
    )
    avg_training_score: float = Field(
        0.0, ge=0, le=100, description="Average score on completed training modules"
    )
    overtime_hours: float = Field(0, ge=0, description="Average monthly overtime hours")
    goals_met_avg: float = Field(
        1.0, ge=0, le=1, description="Average fraction of goals met"
    )
    leadership_score: float = Field(3.0, ge=0, le=5, description="Leadership competency score")
    communication_score: float = Field(
        3.0, ge=0, le=5, description="Communication competency score"
    )
    innovation_score: float = Field(3.0, ge=0, le=5, description="Innovation competency score")
    satisfaction_score: float = Field(
        3.0, ge=0, le=5, description="Employee satisfaction survey score"
    )
    potential_score: float = Field(
        3.0, ge=0, le=5, description="Estimated potential score for 9-Box placement"
    )


class ShapDriver(BaseModel):
    """A single SHAP feature contribution."""

    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value for this employee")
    impact: float = Field(..., description="SHAP value (magnitude of impact)")
    direction: str = Field(..., description="'positive' or 'negative' impact on prediction")


class PredictionResponse(BaseModel):
    """Response payload for a single employee prediction."""

    predicted_score: float = Field(..., description="Predicted performance score")
    confidence_interval: tuple[float, float] = Field(
        ..., description="Lower and upper bounds of the 90% confidence interval"
    )
    nine_box_label: str = Field(..., description="9-Box Grid classification label")
    top_drivers: list[ShapDriver] = Field(
        default_factory=list, description="Top SHAP feature drivers"
    )
    recommendations: list[dict[str, str]] = Field(
        default_factory=list, description="Prioritised development recommendations"
    )


class BatchPredictionRequest(BaseModel):
    """Request payload for batch predictions."""

    employees: list[EmployeeInput] = Field(
        ..., min_length=1, description="List of employee inputs"
    )


class BatchPredictionResponse(BaseModel):
    """Response payload for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="Prediction results for each employee"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status ('healthy' or 'degraded')")
    version: str = Field(..., description="Application version string")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")
    uptime_seconds: float = Field(..., description="Seconds since service start")


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    model_type: str = Field(..., description="Model algorithm type")
    features: list[str] = Field(..., description="Feature names used by the model")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Training/evaluation metrics"
    )
    training_date: str | None = Field(
        None, description="ISO date when the model was last trained"
    )


class NineBoxRequest(BaseModel):
    """Request payload for 9-Box classification."""

    performance_score: float = Field(..., ge=0, le=5, description="Performance score")
    potential_score: float = Field(..., ge=0, le=5, description="Potential score")


class NineBoxResponse(BaseModel):
    """Response payload for 9-Box classification."""

    box_label: str = Field(..., description="9-Box label (e.g. 'Star', 'Core Player')")
    box_number: int = Field(..., ge=1, le=9, description="Box number (1-9)")
    performance_category: str = Field(..., description="'Low', 'Medium', or 'High'")
    potential_category: str = Field(..., description="'Low', 'Medium', or 'High'")
    recommendation: str = Field(..., description="Development recommendation text")
