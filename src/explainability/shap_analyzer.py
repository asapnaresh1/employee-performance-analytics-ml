"""
SHAP-based model explainability for employee performance predictions.

Provides global and local explanations using TreeExplainer for
tree-based models and LinearExplainer for linear models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]
    logger.warning("shap is not installed; ShapAnalyzer will not be available")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]
    logger.warning("matplotlib is not installed; plotting features disabled")


class ShapAnalyzer:
    """Generate SHAP explanations for a trained model."""

    _TREE_TYPES = {"tree", "forest", "xgboost", "lightgbm", "catboost", "gradient_boosting"}
    _LINEAR_TYPES = {"linear", "logistic", "ridge", "lasso", "elasticnet"}

    def __init__(self, model: Any, model_type: str = "tree") -> None:
        if shap is None:
            raise ImportError("shap must be installed to use ShapAnalyzer")

        self.model = model
        self.model_type = model_type.lower()
        self._explainer: shap.Explainer | None = None

        logger.info("ShapAnalyzer initialised for model_type={}", self.model_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_explainer(self, X: pd.DataFrame) -> shap.Explainer:
        """Lazily create and cache the appropriate SHAP explainer."""
        if self._explainer is not None:
            return self._explainer

        if self.model_type in self._TREE_TYPES:
            self._explainer = shap.TreeExplainer(self.model)
            logger.debug("Created TreeExplainer")
        elif self.model_type in self._LINEAR_TYPES:
            self._explainer = shap.LinearExplainer(self.model, X)
            logger.debug("Created LinearExplainer")
        else:
            self._explainer = shap.Explainer(self.model, X)
            logger.debug("Created generic Explainer for model_type={}", self.model_type)

        return self._explainer

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values as a 2-D numpy array."""
        explainer = self._get_explainer(X)
        sv = explainer.shap_values(X)

        # Some explainers return a list (one array per class).
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]

        return np.asarray(sv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_global(self, X: pd.DataFrame) -> dict[str, Any]:
        """Compute global feature importance based on mean |SHAP|.

        Returns
        -------
        dict
            feature_importance : dict[str, float]
            shap_values : np.ndarray
            expected_value : float
        """
        logger.info("Computing global SHAP analysis on {} samples", len(X))
        explainer = self._get_explainer(X)
        shap_values = self._compute_shap_values(X)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        feature_importance = {
            col: float(imp) for col, imp in zip(X.columns, mean_abs, strict=True)
        }
        # Sort descending by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
        )

        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = float(expected[1]) if len(expected) == 2 else float(expected[0])

        return {
            "feature_importance": feature_importance,
            "shap_values": shap_values,
            "expected_value": float(expected),
        }

    def analyze_local(self, X: pd.DataFrame, index: int = 0) -> dict[str, Any]:
        """Explain a single prediction.

        Returns
        -------
        dict
            feature_contributions : dict[str, float]  (sorted by |value|)
            base_value : float
            predicted_value : float
        """
        logger.info("Computing local SHAP analysis for index={}", index)
        explainer = self._get_explainer(X)
        shap_values = self._compute_shap_values(X)

        row_sv = shap_values[index]
        contributions = {
            col: float(sv) for col, sv in zip(X.columns, row_sv, strict=True)
        }
        contributions = dict(
            sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = float(expected[1]) if len(expected) == 2 else float(expected[0])

        predicted = float(expected) + float(np.sum(row_sv))

        return {
            "feature_contributions": contributions,
            "base_value": float(expected),
            "predicted_value": predicted,
        }

    def plot_summary(
        self, X: pd.DataFrame, output_path: Path | None = None
    ) -> None:
        """Create a SHAP beeswarm summary plot."""
        if plt is None:
            logger.error("matplotlib is required for plotting")
            return

        shap_values = self._compute_shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, show=False)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Summary plot saved to {}", output_path)

        plt.close()

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        index: int = 0,
        output_path: Path | None = None,
    ) -> None:
        """Create a SHAP waterfall plot for a single observation."""
        if plt is None:
            logger.error("matplotlib is required for plotting")
            return

        explainer = self._get_explainer(X)
        shap_values = self._compute_shap_values(X)

        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = float(expected[1]) if len(expected) == 2 else float(expected[0])

        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=expected,
            data=X.iloc[index].values,
            feature_names=list(X.columns),
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Waterfall plot saved to {}", output_path)

        plt.close()

    def get_top_drivers(
        self, X: pd.DataFrame, index: int, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Return the top-N feature drivers for a single prediction.

        Each entry contains:
            feature, value, shap_value, direction
        """
        shap_values = self._compute_shap_values(X)
        row_sv = shap_values[index]
        row_data = X.iloc[index]

        pairs = sorted(
            zip(X.columns, row_data, row_sv, strict=True),
            key=lambda t: abs(t[2]),
            reverse=True,
        )

        drivers: list[dict[str, Any]] = []
        for feat, val, sv in pairs[:top_n]:
            drivers.append(
                {
                    "feature": str(feat),
                    "value": float(val),
                    "shap_value": float(sv),
                    "direction": "positive" if float(sv) >= 0 else "negative",
                }
            )

        logger.debug("Top {} drivers for index {}: {}", top_n, index, drivers)
        return drivers
