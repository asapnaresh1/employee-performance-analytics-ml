from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class EmployeeProfileClusterer:
    """K-Means clustering for employee profile segmentation.

    Uses StandardScaler for feature normalisation followed by KMeans.
    """

    FEATURE_COLUMNS: list[str] = [
        "performance_score",
        "potential_score",
        "leadership_score",
        "communication_score",
        "innovation_score",
        "total_training_hours",
        "training_completion_rate",
        "avg_training_score",
        "tenure_years",
        "overtime_avg",
        "goals_met_avg",
    ]

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.model_: KMeans | None = None
        self.cluster_labels_: np.ndarray | None = None

        logger.debug(
            "EmployeeProfileClusterer initialised — n_clusters={}, random_state={}",
            n_clusters,
            random_state,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        if not available:
            raise ValueError(
                "None of the expected feature columns found in DataFrame. "
                f"Expected at least some of: {self.FEATURE_COLUMNS}"
            )
        if len(available) < len(self.FEATURE_COLUMNS):
            missing = set(self.FEATURE_COLUMNS) - set(available)
            logger.warning("Missing feature columns (will be skipped): {}", missing)
        return df[available]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> EmployeeProfileClusterer:
        """Fit scaler and KMeans on *df*.

        Parameters
        ----------
        df:
            DataFrame with at least a subset of :attr:`FEATURE_COLUMNS`.

        Returns
        -------
        self
        """
        features = self._resolve_features(df)
        logger.info(
            "Fitting clusterer on {} samples with {} features",
            len(features),
            features.shape[1],
        )

        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(features)

        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.cluster_labels_ = self.model_.fit_predict(scaled)

        score = silhouette_score(scaled, self.cluster_labels_)
        logger.info("Fit complete — silhouette score: {:.4f}", score)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data.

        Parameters
        ----------
        df:
            DataFrame with the same feature columns used during :meth:`fit`.

        Returns
        -------
        Array of cluster labels.
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Clusterer has not been fitted. Call fit() first.")

        features = self._resolve_features(df)
        scaled = self.scaler_.transform(features)
        labels: np.ndarray = self.model_.predict(scaled)

        logger.info("Predicted clusters for {} samples", len(df))
        return labels

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and return *df* with an added ``cluster`` column.

        Parameters
        ----------
        df:
            DataFrame with at least a subset of :attr:`FEATURE_COLUMNS`.

        Returns
        -------
        Copy of *df* with a new ``cluster`` column.
        """
        self.fit(df)
        out = df.copy()
        out["cluster"] = self.cluster_labels_
        return out

    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return mean feature values per cluster.

        *df* must contain a ``cluster`` column (as produced by
        :meth:`fit_predict`).
        """
        if "cluster" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'cluster' column. "
                "Run fit_predict first."
            )

        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        profiles = (
            df.groupby("cluster")[available]
            .mean()
            .round(3)
            .reset_index()
        )

        logger.info("Cluster profiles computed for {} clusters", len(profiles))
        return profiles

    def get_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: range = range(2, 11),
    ) -> dict[str, Any]:
        """Evaluate silhouette scores for a range of *k* values.

        Parameters
        ----------
        df:
            DataFrame with feature columns.
        k_range:
            Range of cluster counts to evaluate.

        Returns
        -------
        dict with ``silhouette_scores`` (dict[int, float]) and
        ``recommended_k`` (int).
        """
        features = self._resolve_features(df)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        scores: dict[int, float] = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(scaled)
            scores[k] = round(float(silhouette_score(scaled, labels)), 4)
            logger.debug("k={} — silhouette={:.4f}", k, scores[k])

        best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
        logger.info("Optimal k={} (silhouette={:.4f})", best_k, scores[best_k])

        return {
            "silhouette_scores": scores,
            "recommended_k": best_k,
        }
