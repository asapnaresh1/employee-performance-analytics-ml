"""Tests for src.data.generator.PerformanceDataGenerator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import DEPARTMENT_NAMES, PerformanceDataGenerator


class TestPerformanceDataGenerator:
    """Unit tests for the synthetic data generator."""

    def test_employee_count(self, sample_employees: pd.DataFrame) -> None:
        """generate_employees produces exactly n_employees rows."""
        assert len(sample_employees) == 50

    def test_employee_id_unique(self, sample_employees: pd.DataFrame) -> None:
        """Every employee_id is unique."""
        assert sample_employees["employee_id"].is_unique

    def test_employee_columns(self, sample_employees: pd.DataFrame) -> None:
        """Employees table contains all expected columns."""
        expected = {
            "employee_id", "first_name", "last_name", "email", "gender",
            "department", "job_title", "job_level", "hire_date", "age",
            "education", "location", "base_salary", "is_active",
        }
        assert expected.issubset(set(sample_employees.columns))

    def test_departments_valid(self, sample_employees: pd.DataFrame) -> None:
        """All departments belong to the predefined set."""
        assert set(sample_employees["department"].unique()).issubset(set(DEPARTMENT_NAMES))

    def test_salary_positive(self, sample_employees: pd.DataFrame) -> None:
        """All salaries are strictly positive."""
        assert (sample_employees["base_salary"] > 0).all()

    def test_evaluation_periods_present(
        self, sample_evaluations: pd.DataFrame
    ) -> None:
        """At least 2 evaluation periods are generated."""
        assert sample_evaluations["evaluation_period"].nunique() >= 2

    def test_evaluation_scores_bounded(
        self, sample_evaluations: pd.DataFrame
    ) -> None:
        """Performance scores lie within [1.0, 5.0]."""
        scores = sample_evaluations["performance_score"]
        assert scores.min() >= 1.0
        assert scores.max() <= 5.0

    def test_training_statuses_valid(
        self, sample_training: pd.DataFrame
    ) -> None:
        """Training statuses are one of the expected values."""
        valid = {"Completed", "In Progress", "Dropped"}
        assert set(sample_training["status"].unique()).issubset(valid)

    def test_completed_training_has_score(
        self, sample_training: pd.DataFrame
    ) -> None:
        """Completed trainings have a non-null score."""
        completed = sample_training[sample_training["status"] == "Completed"]
        assert completed["score"].notna().all()

    def test_promotion_level_increase(
        self, sample_promotions: pd.DataFrame
    ) -> None:
        """Every promotion has new_level > previous_level."""
        if sample_promotions.empty:
            pytest.skip("No promotions generated for n=50 sample")
        assert (sample_promotions["new_level"] > sample_promotions["previous_level"]).all()

    def test_all_tables_generated(
        self, all_tables: dict[str, pd.DataFrame]
    ) -> None:
        """generate_all returns exactly 4 tables."""
        assert set(all_tables.keys()) == {
            "employees", "evaluations", "training_records", "promotions",
        }
        for name, df in all_tables.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
            assert len(df) > 0 or name == "promotions"

    def test_reproducibility(self) -> None:
        """Same seed produces identical data."""
        gen1 = PerformanceDataGenerator(n_employees=20, random_seed=123)
        gen2 = PerformanceDataGenerator(n_employees=20, random_seed=123)

        tables1 = gen1.generate_all()
        tables2 = gen2.generate_all()

        for key in tables1:
            pd.testing.assert_frame_equal(tables1[key], tables2[key])
