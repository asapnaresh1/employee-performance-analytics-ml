"""
Causal inference module for employee performance analytics.

Uses DoWhy for causal modelling and estimation, with refutation
tests to validate the robustness of identified causal effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import dowhy
    from dowhy import CausalModel
except ImportError:
    dowhy = None  # type: ignore[assignment]
    CausalModel = None  # type: ignore[assignment,misc]
    logger.warning("dowhy is not installed; CausalAnalyzer will not be available")

try:
    from econml.dml import LinearDML  # noqa: F401
except ImportError:
    LinearDML = None  # type: ignore[assignment,misc]
    logger.warning("econml is not installed; LinearDML estimator unavailable")

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Predefined causal hypotheses
# ---------------------------------------------------------------------------

_DEFAULT_HYPOTHESES: list[dict[str, Any]] = [
    {
        "name": "Training hours improve performance",
        "treatment": "total_training_hours",
        "outcome": "performance_score",
        "expected_direction": "positive",
    },
    {
        "name": "Excessive overtime reduces performance",
        "treatment": "overtime_hours",
        "outcome": "performance_score",
        "expected_direction": "negative",
    },
    {
        "name": "Manager rating predicts future performance",
        "treatment": "manager_rating",
        "outcome": "performance_score",
        "expected_direction": "positive",
    },
    {
        "name": "Tenure influences performance",
        "treatment": "tenure_years",
        "outcome": "performance_score",
        "expected_direction": "positive",
    },
]


class CausalAnalyzer:
    """Estimate causal effects between workplace factors and performance."""

    def __init__(self, random_seed: int = 42) -> None:
        if dowhy is None:
            raise ImportError("dowhy must be installed to use CausalAnalyzer")

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        logger.info("CausalAnalyzer initialised with seed={}", self.random_seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_confounders(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str] | None,
    ) -> list[str]:
        """Return an explicit confounder list, inferring from columns if needed."""
        if confounders is not None:
            return [c for c in confounders if c in data.columns]

        excluded = {treatment, outcome}
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        inferred = [c for c in numeric_cols if c not in excluded]
        logger.debug("Inferred {} confounders from data", len(inferred))
        return inferred

    @staticmethod
    def _build_graph(
        treatment: str, outcome: str, confounders: list[str]
    ) -> str:
        """Build a DOT-format causal graph string for DoWhy."""
        edges: list[str] = []
        for c in confounders:
            edges.append(f'"{c}" -> "{treatment}";')
            edges.append(f'"{c}" -> "{outcome}";')
        edges.append(f'"{treatment}" -> "{outcome}";')
        return "digraph { " + " ".join(edges) + " }"

    def _refute(
        self,
        model: Any,
        identified: Any,
        estimate: Any,
    ) -> list[dict[str, Any]]:
        """Run standard refutation tests and return results."""
        refutation_methods = [
            ("random_common_cause", "Add a random common cause"),
            ("placebo_treatment_refuter", "Placebo treatment"),
            ("data_subset_refuter", "Data subset"),
        ]

        results: list[dict[str, Any]] = []
        for method_name, label in refutation_methods:
            try:
                ref = model.refute_estimate(
                    identified,
                    estimate,
                    method_name=method_name,
                    random_seed=self.random_seed,
                )
                results.append(
                    {
                        "method": label,
                        "estimated_effect": float(ref.estimated_effect)
                        if ref.estimated_effect is not None
                        else None,
                        "new_effect": float(ref.new_effect)
                        if ref.new_effect is not None
                        else None,
                        "refutation_result": str(ref),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Refutation '{}' failed: {}", label, exc)
                results.append(
                    {"method": label, "error": str(exc)}
                )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_training_effect(
        self,
        data: pd.DataFrame,
        treatment: str = "total_training_hours",
        outcome: str = "performance_score",
        confounders: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate the causal effect of training on performance.

        Returns
        -------
        dict
            ate, ci_lower, ci_upper, p_value, method, refutation_results
        """
        logger.info(
            "Estimating training effect: {} -> {} ({} rows)",
            treatment,
            outcome,
            len(data),
        )

        confs = self._infer_confounders(data, treatment, outcome, confounders)
        graph = self._build_graph(treatment, outcome, confs)

        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )

        identified = causal_model.identify_effect(proceed_when_unidentifiable=True)

        # Prefer LinearDML when available; fall back to linear regression.
        if LinearDML is not None:
            method = "backdoor.econml.dml.LinearDML"
            try:
                estimate = causal_model.estimate_effect(
                    identified,
                    method_name=method,
                    method_params={
                        "init_params": {"random_state": self.random_seed},
                        "fit_params": {},
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LinearDML failed ({}), falling back to OLS", exc)
                method = "backdoor.linear_regression"
                estimate = causal_model.estimate_effect(
                    identified,
                    method_name=method,
                )
        else:
            method = "backdoor.linear_regression"
            estimate = causal_model.estimate_effect(
                identified,
                method_name=method,
            )

        ate = float(estimate.value)

        # Confidence interval / p-value via bootstrap approximation
        ci_lower, ci_upper, p_value = self._bootstrap_ci(
            data, treatment, outcome, confs, ate
        )

        refutations = self._refute(causal_model, identified, estimate)

        return {
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "method": method,
            "refutation_results": refutations,
        }

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str] | None = None,
        method: str = "backdoor.linear_regression",
    ) -> dict[str, Any]:
        """Generic causal effect estimation.

        Returns
        -------
        dict
            ate, ci_lower, ci_upper, p_value, method, refutation_results
        """
        logger.info(
            "Estimating effect: {} -> {} via {} ({} rows)",
            treatment,
            outcome,
            method,
            len(data),
        )

        confs = self._infer_confounders(data, treatment, outcome, confounders)
        graph = self._build_graph(treatment, outcome, confs)

        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )

        identified = causal_model.identify_effect(proceed_when_unidentifiable=True)
        estimate = causal_model.estimate_effect(identified, method_name=method)

        ate = float(estimate.value)
        ci_lower, ci_upper, p_value = self._bootstrap_ci(
            data, treatment, outcome, confs, ate
        )

        refutations = self._refute(causal_model, identified, estimate)

        return {
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "method": method,
            "refutation_results": refutations,
        }

    def run_hypothesis_tests(
        self, data: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Test predefined causal hypotheses.

        Hypotheses tested:
            1. Training hours -> Performance (positive)
            2. Overtime -> Performance (negative for excess)
            3. Manager rating -> Future performance (positive)
            4. Tenure -> Performance (positive)

        Returns a list of result dictionaries.
        """
        logger.info("Running {} predefined hypothesis tests", len(_DEFAULT_HYPOTHESES))
        results: list[dict[str, Any]] = []

        for hypothesis in _DEFAULT_HYPOTHESES:
            treatment = hypothesis["treatment"]
            outcome = hypothesis["outcome"]

            if treatment not in data.columns or outcome not in data.columns:
                logger.warning(
                    "Skipping '{}': columns not found in data", hypothesis["name"]
                )
                results.append(
                    {
                        "hypothesis": hypothesis["name"],
                        "status": "skipped",
                        "reason": "required columns not in data",
                    }
                )
                continue

            try:
                effect = self.estimate_effect(
                    data=data,
                    treatment=treatment,
                    outcome=outcome,
                )
                direction_match = (
                    (hypothesis["expected_direction"] == "positive" and effect["ate"] > 0)
                    or (hypothesis["expected_direction"] == "negative" and effect["ate"] < 0)
                )

                results.append(
                    {
                        "hypothesis": hypothesis["name"],
                        "treatment": treatment,
                        "outcome": outcome,
                        "ate": effect["ate"],
                        "ci_lower": effect["ci_lower"],
                        "ci_upper": effect["ci_upper"],
                        "p_value": effect["p_value"],
                        "expected_direction": hypothesis["expected_direction"],
                        "observed_direction": "positive" if effect["ate"] >= 0 else "negative",
                        "direction_match": direction_match,
                        "significant": effect["p_value"] < 0.05 if effect["p_value"] is not None else None,
                        "status": "completed",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Hypothesis '{}' failed: {}", hypothesis["name"], exc)
                results.append(
                    {
                        "hypothesis": hypothesis["name"],
                        "status": "error",
                        "error": str(exc),
                    }
                )

        return results

    def summarize_findings(self, results: list[dict[str, Any]]) -> str:
        """Generate a human-readable summary of causal analysis results."""
        lines: list[str] = ["Causal Analysis Summary", "=" * 40, ""]

        for i, res in enumerate(results, start=1):
            name = res.get("hypothesis", f"Analysis {i}")
            status = res.get("status", "unknown")

            if status == "skipped":
                lines.append(f"{i}. {name}: SKIPPED ({res.get('reason', '')})")
                lines.append("")
                continue

            if status == "error":
                lines.append(f"{i}. {name}: ERROR ({res.get('error', '')})")
                lines.append("")
                continue

            ate = res.get("ate", 0.0)
            p_val = res.get("p_value")
            ci_lo = res.get("ci_lower")
            ci_hi = res.get("ci_upper")
            sig = res.get("significant", False)
            direction_ok = res.get("direction_match", None)

            sig_label = "SIGNIFICANT" if sig else "NOT significant"
            direction_label = "as expected" if direction_ok else "CONTRARY to expectation"

            lines.append(f"{i}. {name}")
            lines.append(f"   ATE = {ate:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
            if p_val is not None:
                lines.append(f"   p-value = {p_val:.4f} -> {sig_label}")
            lines.append(f"   Direction: {direction_label}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str],
        point_estimate: float,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
    ) -> tuple[float, float, float | None]:
        """Compute bootstrap confidence interval and approximate p-value.

        Returns (ci_lower, ci_upper, p_value).
        """
        rng = np.random.RandomState(self.random_seed)
        estimates: list[float] = []

        for _ in range(n_bootstrap):
            sample = data.sample(n=len(data), replace=True, random_state=rng.randint(0, 2**31))
            try:
                graph = self._build_graph(treatment, outcome, confounders)
                cm = CausalModel(
                    data=sample,
                    treatment=treatment,
                    outcome=outcome,
                    graph=graph,
                )
                ident = cm.identify_effect(proceed_when_unidentifiable=True)
                est = cm.estimate_effect(
                    ident, method_name="backdoor.linear_regression"
                )
                estimates.append(float(est.value))
            except Exception:  # noqa: BLE001
                continue

        if len(estimates) < 10:
            logger.warning("Too few bootstrap samples ({}); CI unreliable", len(estimates))
            se = abs(point_estimate) * 0.5 if point_estimate != 0 else 0.1
            return (
                point_estimate - 1.96 * se,
                point_estimate + 1.96 * se,
                None,
            )

        arr = np.array(estimates)
        ci_lower = float(np.percentile(arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))

        se = float(np.std(arr, ddof=1))
        if se > 0:
            z = point_estimate / se
            if scipy_stats is not None:
                p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            else:
                p_value = None
        else:
            p_value = None

        return ci_lower, ci_upper, p_value
