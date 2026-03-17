from __future__ import annotations

import json
from typing import Any

import pandas as pd
from loguru import logger


class DevelopmentRecommender:
    """Rule-based development recommendation engine.

    Generates prioritised action items for employees based on their 9-Box
    position, competency scores, training gaps, and performance trend.
    """

    RECOMMENDATION_RULES: list[dict[str, Any]] = [
        # --- 9-Box driven rules ---
        {
            "condition": lambda p: p.get("box_label") == "Under Performer",
            "area": "Performance Improvement",
            "priority": "High",
            "action": (
                "Enrol in a structured performance improvement programme with "
                "weekly check-ins and clear short-term targets."
            ),
            "expected_impact": "Raise performance score by 0.5-1.0 within two quarters.",
        },
        {
            "condition": lambda p: p.get("box_label") == "Development Needed",
            "area": "Skills Development",
            "priority": "High",
            "action": (
                "Assign a senior mentor and create a 90-day development plan "
                "covering core competency gaps."
            ),
            "expected_impact": "Accelerate competency growth and move to Core Player within two quarters.",
        },
        {
            "condition": lambda p: p.get("box_label") == "Star",
            "area": "Retention & Growth",
            "priority": "High",
            "action": (
                "Offer executive mentorship, stretch assignments, and discuss "
                "a leadership fast-track path."
            ),
            "expected_impact": "Strengthen retention and prepare for senior leadership pipeline.",
        },
        {
            "condition": lambda p: p.get("box_label") == "High Potential",
            "area": "Performance Acceleration",
            "priority": "High",
            "action": (
                "Pair with a high-performing peer for knowledge transfer and "
                "assign challenging deliverables."
            ),
            "expected_impact": "Convert high potential into high performance within one to two quarters.",
        },
        {
            "condition": lambda p: p.get("box_label") == "Inconsistent Player",
            "area": "Consistency Coaching",
            "priority": "Medium",
            "action": (
                "Conduct root-cause analysis of performance variability and "
                "establish steady-state goals."
            ),
            "expected_impact": "Reduce performance variance and stabilise output quality.",
        },
        # --- Competency-based rules ---
        {
            "condition": lambda p: p.get("leadership_score", 5) < 2.5,
            "area": "Leadership Development",
            "priority": "Medium",
            "action": (
                "Enrol in a leadership fundamentals workshop and assign team "
                "lead responsibilities on a small project."
            ),
            "expected_impact": "Improve leadership competency score by 0.5-1.0 within one quarter.",
        },
        {
            "condition": lambda p: p.get("communication_score", 5) < 2.5,
            "area": "Communication Skills",
            "priority": "Medium",
            "action": (
                "Participate in presentation skills training and join "
                "cross-functional collaboration initiatives."
            ),
            "expected_impact": "Enhance communication effectiveness and stakeholder feedback ratings.",
        },
        {
            "condition": lambda p: p.get("innovation_score", 5) < 2.5,
            "area": "Innovation & Creativity",
            "priority": "Medium",
            "action": (
                "Attend design-thinking workshops and allocate 10%% time to "
                "innovation sprints or hackathons."
            ),
            "expected_impact": "Increase innovation contributions and idea pipeline quality.",
        },
        # --- Training gap rules ---
        {
            "condition": lambda p: p.get("training_completion_rate", 1.0) < 0.6,
            "area": "Training Completion",
            "priority": "High",
            "action": (
                "Review current training load, remove blockers, and set "
                "bi-weekly completion milestones."
            ),
            "expected_impact": "Raise training completion rate above 80%% within one quarter.",
        },
        {
            "condition": lambda p: p.get("total_training_hours", 100) < 20,
            "area": "Learning Engagement",
            "priority": "Medium",
            "action": (
                "Create a personalised learning path and dedicate protected "
                "hours for professional development."
            ),
            "expected_impact": "Increase training hours and close skill gaps within two quarters.",
        },
        # --- Performance trend rules ---
        {
            "condition": lambda p: p.get("performance_trend", 0) < -0.1,
            "area": "Trend Reversal",
            "priority": "High",
            "action": (
                "Schedule a career development conversation to identify "
                "disengagement factors and co-create an action plan."
            ),
            "expected_impact": "Reverse declining trend and stabilise performance within one quarter.",
        },
        {
            "condition": lambda p: p.get("goals_met_avg", 1.0) < 0.5,
            "area": "Goal Achievement",
            "priority": "High",
            "action": (
                "Revisit goal-setting process with SMART criteria and "
                "implement monthly progress reviews."
            ),
            "expected_impact": "Improve goal completion rate above 70%% within two quarters.",
        },
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, employee_profile: dict[str, Any]) -> list[dict[str, str]]:
        """Generate development recommendations for a single employee.

        Parameters
        ----------
        employee_profile:
            Dictionary with keys such as ``box_label``, ``leadership_score``,
            ``communication_score``, ``innovation_score``,
            ``training_completion_rate``, ``total_training_hours``,
            ``performance_trend``, ``goals_met_avg``.

        Returns
        -------
        List of recommendation dicts, each with keys: ``area``, ``priority``,
        ``action``, ``expected_impact``.
        """
        recommendations: list[dict[str, str]] = []

        for rule in self.RECOMMENDATION_RULES:
            try:
                if rule["condition"](employee_profile):
                    recommendations.append(
                        {
                            "area": rule["area"],
                            "priority": rule["priority"],
                            "action": rule["action"],
                            "expected_impact": rule["expected_impact"],
                        }
                    )
            except Exception:  # noqa: BLE001
                continue

        # Sort: High > Medium > Low
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 99))

        logger.debug(
            "Generated {} recommendations for employee profile",
            len(recommendations),
        )
        return recommendations

    def recommend_batch(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate top-3 recommendations for every row in *df*.

        Each row is treated as an employee profile dictionary. A new column
        ``recommendations`` is added containing the top-3 recommendations
        serialised as a JSON string.

        Parameters
        ----------
        df:
            DataFrame where each row represents an employee with relevant
            profile fields.

        Returns
        -------
        Copy of *df* with an added ``recommendations`` column.
        """
        logger.info("Generating batch recommendations for {} employees", len(df))

        def _top3(row: pd.Series) -> str:
            profile = row.to_dict()
            recs = self.recommend(profile)[:3]
            return json.dumps(recs, ensure_ascii=False)

        out = df.copy()
        out["recommendations"] = df.apply(_top3, axis=1)

        logger.info("Batch recommendations complete")
        return out
