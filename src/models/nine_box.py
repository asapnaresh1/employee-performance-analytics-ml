from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class NineBoxClassifier:
    """9-Box Grid classification for talent management.

    Maps employees into a 3x3 matrix based on performance and potential scores,
    producing actionable labels and development recommendations.
    """

    def __init__(
        self,
        perf_thresholds: tuple[float, float] = (2.5, 3.5),
        pot_thresholds: tuple[float, float] = (2.5, 3.5),
    ) -> None:
        self.perf_low, self.perf_high = perf_thresholds
        self.pot_low, self.pot_high = pot_thresholds

        self.BOX_LABELS: dict[tuple[str, str], str] = {
            ("High", "High"): "Star",
            ("High", "Medium"): "High Performer",
            ("High", "Low"): "Solid Performer",
            ("Medium", "High"): "High Potential",
            ("Medium", "Medium"): "Core Player",
            ("Medium", "Low"): "Average Performer",
            ("Low", "High"): "Inconsistent Player",
            ("Low", "Medium"): "Development Needed",
            ("Low", "Low"): "Under Performer",
        }

        self.BOX_NUMBERS: dict[str, int] = {
            "Star": 1,
            "High Performer": 2,
            "Solid Performer": 3,
            "High Potential": 4,
            "Core Player": 5,
            "Average Performer": 6,
            "Inconsistent Player": 7,
            "Development Needed": 8,
            "Under Performer": 9,
        }

        self.BOX_RECOMMENDATIONS: dict[str, str] = {
            "Star": (
                "Invest heavily in retention. Offer stretch assignments, executive "
                "mentoring, and leadership fast-track programmes."
            ),
            "High Performer": (
                "Recognise contributions publicly. Provide cross-functional exposure "
                "and explore potential-unlocking coaching."
            ),
            "Solid Performer": (
                "Maintain engagement with meaningful work. Explore lateral moves "
                "to broaden experience and reignite growth."
            ),
            "High Potential": (
                "Accelerate development with targeted training, challenging projects, "
                "and a performance improvement plan with clear milestones."
            ),
            "Core Player": (
                "Encourage skill deepening. Pair with mentors and set incremental "
                "performance goals to move towards higher boxes."
            ),
            "Average Performer": (
                "Identify specific skill gaps. Provide structured training and "
                "quarterly check-ins to track improvement."
            ),
            "Inconsistent Player": (
                "Investigate root causes of inconsistency. Provide coaching and a "
                "focused performance improvement plan with short-term targets."
            ),
            "Development Needed": (
                "Assign a dedicated mentor. Create a 90-day development plan with "
                "measurable outcomes and regular feedback sessions."
            ),
            "Under Performer": (
                "Initiate a formal performance improvement plan. Provide clear "
                "expectations, weekly reviews, and consider role realignment."
            ),
        }

        logger.debug(
            "NineBoxClassifier initialised — perf thresholds={}, pot thresholds={}",
            perf_thresholds,
            pot_thresholds,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _categorise(self, value: float, low: float, high: float) -> str:
        if value < low:
            return "Low"
        if value < high:
            return "Medium"
        return "High"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        performance_score: float,
        potential_score: float,
    ) -> dict[str, Any]:
        """Classify a single employee into the 9-Box Grid.

        Parameters
        ----------
        performance_score:
            Numeric performance rating.
        potential_score:
            Numeric potential rating.

        Returns
        -------
        dict with keys:
            box_label, box_number (1-9), performance_category, potential_category,
            recommendation.
        """
        perf_cat = self._categorise(performance_score, self.perf_low, self.perf_high)
        pot_cat = self._categorise(potential_score, self.pot_low, self.pot_high)

        label = self.BOX_LABELS[(perf_cat, pot_cat)]
        box_number = self.BOX_NUMBERS[label]
        recommendation = self.BOX_RECOMMENDATIONS[label]

        return {
            "box_label": label,
            "box_number": box_number,
            "performance_category": perf_cat,
            "potential_category": pot_cat,
            "recommendation": recommendation,
        }

    def classify_batch(
        self,
        df: pd.DataFrame,
        perf_col: str = "performance_score",
        pot_col: str = "potential_score",
    ) -> pd.DataFrame:
        """Classify every row in *df* and append 9-Box columns.

        Parameters
        ----------
        df:
            DataFrame containing at least *perf_col* and *pot_col*.
        perf_col:
            Column name for performance scores.
        pot_col:
            Column name for potential scores.

        Returns
        -------
        A copy of *df* with added columns: ``box_label``, ``box_number``,
        ``performance_category``, ``potential_category``, ``recommendation``.
        """
        logger.info("Classifying {} employees into the 9-Box Grid", len(df))

        results = df.apply(
            lambda row: self.classify(row[perf_col], row[pot_col]),
            axis=1,
            result_type="expand",
        )

        out = df.copy()
        for col in results.columns:
            out[col] = results[col].values

        logger.info("Classification complete")
        return out

    def get_distribution(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return counts and percentages for each box present in *df*.

        *df* must already contain a ``box_label`` column (as produced by
        :meth:`classify_batch`).
        """
        if "box_label" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'box_label' column. "
                "Run classify_batch first."
            )

        counts = df["box_label"].value_counts().reset_index()
        counts.columns = ["box_label", "count"]
        counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
        counts["box_number"] = counts["box_label"].map(self.BOX_NUMBERS)
        counts = counts.sort_values("box_number").reset_index(drop=True)

        logger.info("Distribution across {} boxes computed", len(counts))
        return counts
