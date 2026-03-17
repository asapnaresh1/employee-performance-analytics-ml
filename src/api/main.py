"""FastAPI application factory with lifespan-managed model loading."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import get_settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artefacts on startup and clean up on shutdown."""
    app.state.start_time = time.time()
    app.state.model = None
    app.state.shap_analyzer = None
    app.state.model_type = "unknown"
    app.state.model_metrics = {}  # type: ignore[assignment]
    app.state.training_date = None

    model_path = _MODELS_DIR / "predictor.joblib"
    metrics_path = _MODELS_DIR / "metrics.joblib"

    if model_path.exists():
        try:
            artefact: dict[str, Any] = joblib.load(model_path)
            app.state.model = artefact["model"]
            app.state.model_type = artefact.get("model_type", "unknown")
            logger.info(
                "Model loaded from {} (type={})", model_path, app.state.model_type
            )

            if metrics_path.exists():
                app.state.model_metrics = joblib.load(metrics_path)
                logger.info("Model metrics loaded from {}", metrics_path)

            # Attempt to create SHAP explainer
            try:
                from src.explainability.shap_analyzer import ShapAnalyzer

                model_type_map: dict[str, str] = {
                    "xgboost": "xgboost",
                    "random_forest": "forest",
                    "gradient_boosting": "gradient_boosting",
                }
                shap_type = model_type_map.get(app.state.model_type, "tree")
                app.state.shap_analyzer = ShapAnalyzer(
                    app.state.model, model_type=shap_type
                )
                logger.info("SHAP explainer initialised")
            except Exception:
                logger.warning(
                    "SHAP explainer could not be initialised; "
                    "predictions will be returned without feature explanations"
                )
        except Exception:
            logger.error(
                "Failed to load model from {}; starting in degraded mode",
                model_path,
            )
    else:
        logger.warning(
            "Model file not found at {}; starting in degraded mode", model_path
        )

    yield

    logger.info("Shutting down — cleaning up resources")
    app.state.model = None
    app.state.shap_analyzer = None


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "REST API for employee performance prediction, 9-Box classification, "
            "and development recommendations powered by tree-based ML models."
        ),
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.routes import router

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
