from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


class PerformancePredictor:
    """Predict future employee performance using tree-based regressors.

    Supported model types: ``"xgboost"``, ``"random_forest"``,
    ``"gradient_boosting"``.
    """

    FEATURE_COLUMNS: list[str] = [
        "avg_past_performance",
        "performance_trend",
        "total_training_hours",
        "training_completion_rate",
        "avg_training_score",
        "tenure_years",
        "job_level",
        "department_encoded",
        "overtime_avg",
        "goals_met_avg",
        "leadership_score",
        "communication_score",
        "innovation_score",
    ]

    _MODEL_MAP: dict[str, type] = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state

        self.model_: Any | None = None
        self.feature_importances_: np.ndarray | None = None

        logger.debug(
            "PerformancePredictor initialised — model_type='{}', random_state={}",
            model_type,
            random_state,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0,
            )

        cls = self._MODEL_MAP.get(self.model_type)
        if cls is None:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Choose from: 'xgboost', 'random_forest', 'gradient_boosting'."
            )
        return cls(
            n_estimators=200,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_features(
        evaluations: pd.DataFrame,
        training: pd.DataFrame,
        employees: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge and engineer features from raw tables.

        Parameters
        ----------
        evaluations:
            Historical evaluation records with columns such as
            ``employee_id``, ``performance_score``, ``evaluation_date``,
            ``goals_met``, ``overtime_hours``.
        training:
            Training records with ``employee_id``, ``training_hours``,
            ``completion_status``, ``score``.
        employees:
            Employee master data with ``employee_id``, ``hire_date``,
            ``job_level``, ``department``.

        Returns
        -------
        A DataFrame with one row per employee and engineered feature columns.
        """
        logger.info("Preparing features from evaluation, training and employee data")

        # --- Performance aggregates ---
        perf_agg = (
            evaluations.groupby("employee_id")["performance_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_past_performance", "count": "eval_count"})
            .reset_index()
        )

        # Performance trend (simple slope via rank correlation)
        def _slope(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return 0.0
            x = np.arange(len(group), dtype=float)
            y = group.sort_values("evaluation_date")["performance_score"].values.astype(float)
            coef: np.ndarray = np.polyfit(x, y, 1)
            return float(coef[0])

        trends = (
            evaluations.groupby("employee_id")
            .apply(_slope, include_groups=False)
            .reset_index(name="performance_trend")
        )

        # Goals met average
        goals = pd.DataFrame()
        if "goals_met" in evaluations.columns:
            goals = (
                evaluations.groupby("employee_id")["goals_met"]
                .mean()
                .reset_index(name="goals_met_avg")
            )

        # Overtime average
        overtime = pd.DataFrame()
        if "overtime_hours" in evaluations.columns:
            overtime = (
                evaluations.groupby("employee_id")["overtime_hours"]
                .mean()
                .reset_index(name="overtime_avg")
            )

        # --- Training aggregates ---
        train_hours = (
            training.groupby("employee_id")["training_hours"]
            .sum()
            .reset_index(name="total_training_hours")
        )

        train_completion = pd.DataFrame()
        if "completion_status" in training.columns:
            train_completion = (
                training.groupby("employee_id")["completion_status"]
                .mean()
                .reset_index(name="training_completion_rate")
            )

        train_score = pd.DataFrame()
        if "score" in training.columns:
            train_score = (
                training.groupby("employee_id")["score"]
                .mean()
                .reset_index(name="avg_training_score")
            )

        # --- Employee master ---
        emp = employees[["employee_id"]].copy()

        if "hire_date" in employees.columns:
            emp["tenure_years"] = (
                (pd.Timestamp.now() - pd.to_datetime(employees["hire_date"])).dt.days
                / 365.25
            ).round(2)

        if "job_level" in employees.columns:
            emp["job_level"] = employees["job_level"]

        if "department" in employees.columns:
            emp["department_encoded"] = employees["department"].astype("category").cat.codes

        # Competency scores if available
        for col in ("leadership_score", "communication_score", "innovation_score"):
            if col in employees.columns:
                emp[col] = employees[col]

        # --- Merge ---
        features = emp.copy()
        for right_df in (
            perf_agg,
            trends,
            goals,
            overtime,
            train_hours,
            train_completion,
            train_score,
        ):
            if not right_df.empty and "employee_id" in right_df.columns:
                features = features.merge(right_df, on="employee_id", how="left")

        features = features.fillna(0)

        logger.info(
            "Feature preparation complete — {} employees, {} features",
            len(features),
            features.shape[1] - 1,
        )
        return features

    # ------------------------------------------------------------------
    # Training / prediction
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the model and return in-sample metrics.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target variable (performance score to predict).

        Returns
        -------
        dict with R2, MAE, RMSE.
        """
        logger.info(
            "Training {} on {} samples with {} features",
            self.model_type,
            len(X),
            X.shape[1],
        )

        self.model_ = self._build_model()
        self.model_.fit(X, y)

        self.feature_importances_ = np.array(self.model_.feature_importances_)

        preds = self.model_.predict(X)
        metrics = {
            "r2": round(float(r2_score(y, preds)), 4),
            "mae": round(float(mean_absolute_error(y, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
        }

        logger.info("Training metrics: {}", metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for *X*.

        Parameters
        ----------
        X:
            Feature matrix (same columns as used in :meth:`fit`).

        Returns
        -------
        Array of predicted performance scores.
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        predictions: np.ndarray = self.model_.predict(X)
        logger.info("Generated predictions for {} samples", len(X))
        return predictions

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> dict[str, Any]:
        """Run k-fold cross-validation and return per-fold metrics.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target variable.
        cv:
            Number of folds.

        Returns
        -------
        dict with ``fold_r2_scores``, ``mean_r2``, ``std_r2``.
        """
        logger.info("Running {}-fold cross-validation for {}", cv, self.model_type)

        model = self._build_model()
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        result = {
            "fold_r2_scores": [round(float(s), 4) for s in scores],
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        }

        logger.info("CV results — mean R²={:.4f} (±{:.4f})", result["mean_r2"], result["std_r2"])
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist fitted model and feature importances to disk via joblib."""
        if self.model_ is None:
            raise RuntimeError("No fitted model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artefact = {
            "model": self.model_,
            "feature_importances": self.feature_importances_,
            "model_type": self.model_type,
        }
        joblib.dump(artefact, path)
        logger.info("Model saved to {}", path)

    def load(self, path: Path) -> None:
        """Load a previously saved model from *path*."""
        path = Path(path)
        artefact = joblib.load(path)

        self.model_ = artefact["model"]
        self.feature_importances_ = artefact.get("feature_importances")
        self.model_type = artefact.get("model_type", self.model_type)

        logger.info("Model loaded from {}", path)
