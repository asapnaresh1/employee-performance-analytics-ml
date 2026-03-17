"""
Synthetic data generator for employee performance analytics.

Generates four correlated tables (employees, evaluations, training_records,
promotions) with realistic statistical relationships for ML experimentation.

Tables
------
- employees: demographic and employment info (n_employees rows)
- evaluations: periodic performance reviews (n_employees x periods)
- training_records: course completions (variable per employee)
- promotions: career advancement events (subset of employees)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPARTMENTS: dict[str, dict] = {
    "Engineering": {
        "titles": {
            1: "Junior Engineer",
            2: "Engineer",
            3: "Senior Engineer",
            4: "Staff Engineer",
            5: "Principal Engineer",
            6: "Distinguished Engineer",
        },
        "salary_range": (55_000, 220_000),
    },
    "Sales": {
        "titles": {
            1: "Sales Development Rep",
            2: "Account Executive",
            3: "Senior Account Executive",
            4: "Sales Manager",
            5: "Regional Sales Director",
            6: "VP of Sales",
        },
        "salary_range": (45_000, 200_000),
    },
    "Marketing": {
        "titles": {
            1: "Marketing Coordinator",
            2: "Marketing Specialist",
            3: "Senior Marketing Specialist",
            4: "Marketing Manager",
            5: "Senior Marketing Manager",
            6: "VP of Marketing",
        },
        "salary_range": (42_000, 185_000),
    },
    "HR": {
        "titles": {
            1: "HR Assistant",
            2: "HR Generalist",
            3: "Senior HR Generalist",
            4: "HR Manager",
            5: "Senior HR Manager",
            6: "VP of Human Resources",
        },
        "salary_range": (40_000, 175_000),
    },
    "Finance": {
        "titles": {
            1: "Financial Analyst I",
            2: "Financial Analyst II",
            3: "Senior Financial Analyst",
            4: "Finance Manager",
            5: "Senior Finance Manager",
            6: "VP of Finance",
        },
        "salary_range": (50_000, 210_000),
    },
    "Operations": {
        "titles": {
            1: "Operations Associate",
            2: "Operations Analyst",
            3: "Senior Operations Analyst",
            4: "Operations Manager",
            5: "Senior Operations Manager",
            6: "VP of Operations",
        },
        "salary_range": (40_000, 180_000),
    },
    "Product": {
        "titles": {
            1: "Associate Product Manager",
            2: "Product Manager",
            3: "Senior Product Manager",
            4: "Group Product Manager",
            5: "Director of Product",
            6: "VP of Product",
        },
        "salary_range": (55_000, 215_000),
    },
    "Customer Support": {
        "titles": {
            1: "Support Representative",
            2: "Senior Support Representative",
            3: "Support Team Lead",
            4: "Support Manager",
            5: "Senior Support Manager",
            6: "VP of Customer Support",
        },
        "salary_range": (35_000, 160_000),
    },
}

DEPARTMENT_NAMES: list[str] = list(DEPARTMENTS.keys())
DEPARTMENT_WEIGHTS: list[float] = [0.25, 0.15, 0.10, 0.08, 0.10, 0.12, 0.10, 0.10]

EDUCATION_LEVELS: list[str] = ["High School", "Bachelor", "Master", "PhD"]
EDUCATION_WEIGHTS: list[float] = [0.10, 0.45, 0.35, 0.10]
EDUCATION_SALARY_MULT: dict[str, float] = {
    "High School": 0.85,
    "Bachelor": 1.00,
    "Master": 1.12,
    "PhD": 1.25,
}
EDUCATION_PERF_BONUS: dict[str, float] = {
    "High School": -0.15,
    "Bachelor": 0.0,
    "Master": 0.10,
    "PhD": 0.20,
}

GENDER_OPTIONS: list[str] = ["Male", "Female", "Non-Binary"]
GENDER_WEIGHTS: list[float] = [0.48, 0.48, 0.04]

LOCATIONS: list[str] = [
    "New York", "San Francisco", "Chicago", "Austin",
    "Seattle", "Boston", "Denver", "Remote",
]

DEFAULT_EVALUATION_PERIODS: list[str] = [
    "H1-2021", "H2-2021", "H1-2022", "H2-2022",
    "H1-2023", "H2-2023", "H1-2024", "H2-2024",
]

PERIOD_START_DATES: dict[str, str] = {
    "H1-2021": "2021-01-15", "H2-2021": "2021-07-15",
    "H1-2022": "2022-01-15", "H2-2022": "2022-07-15",
    "H1-2023": "2023-01-15", "H2-2023": "2023-07-15",
    "H1-2024": "2024-01-15", "H2-2024": "2024-07-15",
}

TRAINING_CATEGORIES: list[str] = [
    "Technical", "Leadership", "Communication", "Compliance", "Domain",
]

TRAINING_NAMES: dict[str, list[str]] = {
    "Technical": [
        "Python for Data Science", "Advanced Machine Learning",
        "Cloud Architecture (AWS)", "SQL & Data Warehousing",
        "Cybersecurity Fundamentals", "DevOps Practices",
        "Microservices Design Patterns", "React & Modern Frontend",
    ],
    "Leadership": [
        "Leadership Essentials", "Managing High-Performing Teams",
        "Strategic Thinking Workshop", "Coaching & Mentoring",
        "Agile Project Management",
    ],
    "Communication": [
        "Effective Communication", "Presentation Skills",
        "Negotiation Tactics", "Conflict Resolution",
        "Business Writing Excellence",
    ],
    "Compliance": [
        "Compliance & Ethics", "Diversity & Inclusion",
        "Data Privacy (GDPR/CCPA)", "Workplace Safety",
        "Anti-Harassment Training",
    ],
    "Domain": [
        "Financial Modelling", "Product Strategy",
        "Design Thinking", "Market Research Methods",
        "Customer Success Fundamentals",
    ],
}

DEPARTMENT_PERF_OFFSET: dict[str, float] = {
    "Engineering": 0.10,
    "Product": 0.05,
    "Finance": 0.05,
    "Sales": -0.05,
    "Marketing": 0.0,
    "HR": 0.0,
    "Operations": -0.05,
    "Customer Support": -0.10,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _compute_salary(
    dept: str,
    level: int,
    education: str,
    rng: np.random.Generator,
) -> float:
    """Compute base salary correlated with department, level, and education."""
    info = DEPARTMENTS[dept]
    lo, hi = info["salary_range"]
    # Level proportion across the 1-6 range
    level_frac = (level - 1) / 5.0
    base = lo + (hi - lo) * level_frac
    # Apply education multiplier
    base *= EDUCATION_SALARY_MULT[education]
    # Add noise (+/- 8%)
    noise = rng.normal(1.0, 0.04)
    return round(float(base * noise), 2)


def _period_to_date(period: str, rng: np.random.Generator) -> pd.Timestamp:
    """Convert an evaluation period like 'H1-2023' to a plausible eval date."""
    half, year_str = period.split("-")
    year_int = int(year_str)
    if half == "H1":
        month = int(rng.integers(6, 8))  # June or July
    else:
        month = int(rng.integers(12, 13))  # December
        if month == 13:
            month = 12
    day = int(rng.integers(1, 29))
    return pd.Timestamp(year=year_int, month=month, day=day)


def _period_start_ts(period: str) -> pd.Timestamp:
    """Return the start timestamp of a given evaluation period."""
    half, year_str = period.split("-")
    year_int = int(year_str)
    month = 1 if half == "H1" else 7
    return pd.Timestamp(year=year_int, month=month, day=1)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class PerformanceDataGenerator:
    """Generates correlated synthetic HR / performance data.

    Produces four interconnected DataFrames with realistic statistical
    patterns suitable for ML modelling and analytics dashboards.

    Parameters
    ----------
    n_employees : int
        Number of employees to generate (default ``1500``).
    evaluation_periods : list[str] | None
        Ordered list of half-year period labels (e.g. ``["H1-2021", ...]``).
        Defaults to eight periods from H1-2021 through H2-2024.
    random_seed : int
        Seed for ``numpy`` and ``Faker`` reproducibility.
    """

    def __init__(
        self,
        n_employees: int = 1500,
        evaluation_periods: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.n_employees = n_employees
        self.evaluation_periods = evaluation_periods or DEFAULT_EVALUATION_PERIODS
        self.random_seed = random_seed

        self.rng = np.random.default_rng(random_seed)
        # Create a fully isolated Faker instance with deterministic seed
        self.fake = Faker()
        self.fake.seed_instance(random_seed)
        self.fake.unique.clear()

        # Will be populated by generate methods
        self._employees: pd.DataFrame | None = None
        self._evaluations: pd.DataFrame | None = None
        self._training_records: pd.DataFrame | None = None
        self._promotions: pd.DataFrame | None = None

        # Internal: per-employee latent ability drives cross-table correlations
        self._latent_ability: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_employees(self) -> pd.DataFrame:
        """Generate the employees table."""
        logger.info("Generating employees table ({} rows) ...", self.n_employees)
        n = self.n_employees

        # Latent ability drawn once, reused across all tables
        self._latent_ability = self.rng.normal(0.0, 1.0, size=n)

        genders = self.rng.choice(GENDER_OPTIONS, size=n, p=GENDER_WEIGHTS)
        first_names: list[str] = []
        last_names: list[str] = []
        for g in genders:
            if g == "Male":
                first_names.append(self.fake.first_name_male())
            elif g == "Female":
                first_names.append(self.fake.first_name_female())
            else:
                first_names.append(self.fake.first_name())
            last_names.append(self.fake.last_name())

        departments = self.rng.choice(DEPARTMENT_NAMES, size=n, p=DEPARTMENT_WEIGHTS)

        # Job level correlated with latent ability
        raw_levels = 1 + self.rng.poisson(lam=1.2, size=n) + (self._latent_ability * 0.4).astype(int)
        job_levels: np.ndarray = np.clip(raw_levels, 1, 6)

        # Job titles derived from department + level
        job_titles = [
            DEPARTMENTS[dept]["titles"][int(lvl)]
            for dept, lvl in zip(departments, job_levels, strict=True)
        ]

        # Hire dates: 2010-01-01 to 2023-12-31
        start_ts = int(pd.Timestamp("2010-01-01").timestamp())
        end_ts = int(pd.Timestamp("2023-12-31").timestamp())
        hire_timestamps = self.rng.integers(start_ts, end_ts, size=n)
        hire_dates = pd.to_datetime(hire_timestamps, unit="s").normalize()

        ages = self.rng.integers(22, 61, size=n)
        educations = self.rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_WEIGHTS)
        locations = self.rng.choice(LOCATIONS, size=n)

        # Base salary correlated with department + level + education
        base_salaries = np.array([
            _compute_salary(dept, int(lvl), edu, self.rng)
            for dept, lvl, edu in zip(departments, job_levels, educations, strict=True)
        ])

        # Emails: firstname.lastname@company.com (deduplicated with id suffix)
        emails = [
            f"{fn.lower()}.{ln.lower()}.{eid}@company.com"
            for eid, fn, ln in zip(range(1, n + 1), first_names, last_names, strict=True)
        ]

        # 85% active
        is_active = self.rng.choice([True, False], size=n, p=[0.85, 0.15])

        employee_ids = [f"EMP-{i:04d}" for i in range(1, n + 1)]

        self._employees = pd.DataFrame({
            "employee_id": employee_ids,
            "first_name": first_names,
            "last_name": last_names,
            "email": emails,
            "gender": genders,
            "department": departments,
            "job_title": job_titles,
            "job_level": job_levels,
            "hire_date": hire_dates,
            "age": ages,
            "education": educations,
            "location": locations,
            "base_salary": base_salaries,
            "is_active": is_active,
        })

        logger.info("Employees table: {} rows, {} columns", *self._employees.shape)
        return self._employees

    def generate_evaluations(self, employees: pd.DataFrame) -> pd.DataFrame:
        """Generate the evaluations table with realistic performance patterns.

        Parameters
        ----------
        employees : pd.DataFrame
            The employees table (must have been generated first).
        """
        logger.info("Generating evaluations table ...")
        n = len(employees)
        rows: list[dict] = []
        eval_counter = 0

        hire_dates = pd.to_datetime(employees["hire_date"])
        education_arr = employees["education"].values
        department_arr = employees["department"].values
        job_level_arr = employees["job_level"].values.astype(float)
        is_active_arr = employees["is_active"].values

        # Track per-employee previous score for temporal autocorrelation
        prev_scores = np.full(n, np.nan)

        for period_idx, period in enumerate(self.evaluation_periods):
            period_start = _period_start_ts(period)

            # Employees must be hired before the period and be active
            active_mask = (hire_dates < period_start) & is_active_arr
            active_ids = np.where(active_mask)[0]

            if len(active_ids) == 0:
                continue

            m = len(active_ids)

            # Tenure bonus: years since hire, capped contribution
            tenure_years = (period_start - hire_dates.iloc[active_ids]).dt.days.values / 365.25
            tenure_bonus = np.clip(tenure_years * 0.025, 0.0, 0.30)

            # Level bonus: higher levels have higher base performance
            level_bonus = (job_level_arr[active_ids] - 1) * 0.12

            # Education bonus
            edu_bonus = np.array([EDUCATION_PERF_BONUS[e] for e in education_arr[active_ids]])

            # Department offset
            dept_bonus = np.array([DEPARTMENT_PERF_OFFSET[d] for d in department_arr[active_ids]])

            # Latent ability contribution
            ability = self._latent_ability[active_ids] * 0.40  # type: ignore[index]

            # Time trend: slight average improvement each period
            time_trend = period_idx * 0.02

            # Noise
            noise = self.rng.normal(0.0, 0.30, size=m)

            # Composite raw performance
            raw_perf = 3.0 + ability + level_bonus + tenure_bonus + edu_bonus + dept_bonus + time_trend + noise

            # Temporal autocorrelation: blend with previous score if exists
            prev = prev_scores[active_ids]
            has_prev = ~np.isnan(prev)
            if has_prev.any():
                raw_perf[has_prev] = 0.55 * raw_perf[has_prev] + 0.45 * prev[has_prev]

            performance_score = np.clip(np.round(raw_perf, 2), 1.0, 5.0)

            # --- Correlated sub-scores ---

            # Potential: correlated with performance + ability, distinct noise
            pot_noise = self.rng.normal(0.0, 0.45, size=m)
            potential_score = np.clip(
                np.round(0.45 * performance_score + 0.35 * self._latent_ability[active_ids] + 1.5 + pot_noise, 2),  # type: ignore[index]
                1.0, 5.0,
            )

            # Goals met percentage
            goal_noise = self.rng.normal(0.0, 10.0, size=m)
            goals_met_pct = np.clip(
                np.round((performance_score - 1.0) / 4.0 * 60.0 + 30.0 + goal_noise, 1),
                0.0, 100.0,
            )

            # Manager rating: slight positive bias over actual performance
            mgr_noise = self.rng.normal(0.10, 0.30, size=m)
            manager_rating = np.clip(np.round(performance_score + mgr_noise, 2), 1.0, 5.0)

            # Peer rating: close to actual with moderate noise
            peer_noise = self.rng.normal(0.0, 0.35, size=m)
            peer_rating = np.clip(np.round(performance_score + peer_noise, 2), 1.0, 5.0)

            # Self rating: tends to be higher than manager rating
            self_noise = self.rng.normal(0.30, 0.35, size=m)
            self_rating = np.clip(np.round(performance_score + self_noise, 2), 1.0, 5.0)

            # Overtime hours: moderate for mid-performers, high for top/bottom
            base_ot = 10.0 + np.abs(performance_score - 3.0) * 8.0
            ot_noise = self.rng.exponential(5.0, size=m)
            overtime_hours = np.clip(np.round(base_ot + ot_noise, 0), 0, 80).astype(int)

            # Projects completed: correlated with performance
            proj_base = (performance_score - 1.0) / 4.0 * 6.0 + 1.0
            proj_noise = self.rng.poisson(1, size=m).astype(float)
            projects_completed = np.clip(np.round(proj_base + proj_noise, 0), 0, 10).astype(int)

            # Innovation, communication, leadership scores
            def _sub_score(base: np.ndarray, sigma: float = 0.40) -> np.ndarray:
                sub_noise = self.rng.normal(0.0, sigma, size=len(base))
                return np.clip(np.round(base + sub_noise, 2), 1.0, 5.0)

            innovation_score = _sub_score(performance_score, 0.50)
            communication_score = _sub_score(performance_score, 0.40)
            leadership_score = _sub_score(performance_score, 0.45)

            # Evaluation date: a realistic date within the period
            eval_dates = [_period_to_date(period, self.rng) for _ in range(m)]

            for i, idx in enumerate(active_ids):
                eval_counter += 1
                rows.append({
                    "evaluation_id": f"EVAL-{eval_counter:06d}",
                    "employee_id": employees.iloc[idx]["employee_id"],
                    "evaluation_period": period,
                    "evaluation_date": eval_dates[i],
                    "performance_score": float(performance_score[i]),
                    "potential_score": float(potential_score[i]),  # type: ignore[index]
                    "goals_met_pct": float(goals_met_pct[i]),
                    "manager_rating": float(manager_rating[i]),
                    "peer_rating": float(peer_rating[i]),
                    "self_rating": float(self_rating[i]),
                    "overtime_hours": int(overtime_hours[i]),
                    "projects_completed": int(projects_completed[i]),
                    "innovation_score": float(innovation_score[i]),
                    "communication_score": float(communication_score[i]),
                    "leadership_score": float(leadership_score[i]),
                })

            # Store for next-period autocorrelation
            prev_scores[active_ids] = performance_score

        self._evaluations = pd.DataFrame(rows)
        logger.info("Evaluations table: {} rows, {} columns", *self._evaluations.shape)
        return self._evaluations

    def generate_training(self, employees: pd.DataFrame) -> pd.DataFrame:
        """Generate the training_records table.

        Produces 2-6 training records per employee per active year, with
        completion rates and scores correlated with latent ability.

        Parameters
        ----------
        employees : pd.DataFrame
            The employees table.
        """
        logger.info("Generating training_records table ...")
        rows: list[dict] = []
        tr_counter = 0

        hire_dates = pd.to_datetime(employees["hire_date"])
        n = len(employees)
        all_categories = TRAINING_CATEGORIES

        for emp_idx in range(n):
            emp_id = employees.iloc[emp_idx]["employee_id"]
            hire = hire_dates.iloc[emp_idx]
            ability = self._latent_ability[emp_idx]  # type: ignore[index]

            # Determine active years for this employee
            first_year = max(hire.year, 2021)
            last_year = 2024

            for year in range(first_year, last_year + 1):
                # 2-6 trainings per year, higher ability -> slightly more
                n_trainings = int(np.clip(self.rng.poisson(3.5 + ability * 0.5), 2, 6))

                for _ in range(n_trainings):
                    category = str(self.rng.choice(all_categories))
                    course_list = TRAINING_NAMES[category]
                    training_name = str(self.rng.choice(course_list))

                    # Start date: random within the year (but after hire)
                    year_start = max(
                        pd.Timestamp(year=year, month=1, day=1),
                        hire,
                    )
                    year_end = pd.Timestamp(year=year, month=12, day=28)
                    if year_start >= year_end:
                        continue

                    start_ts = int(year_start.timestamp())
                    end_ts_bound = int(year_end.timestamp())
                    s_ts = self.rng.integers(start_ts, end_ts_bound)
                    start_date = pd.Timestamp(s_ts, unit="s").normalize()

                    # Training duration: 4-40 hours spread over days
                    hours = int(self.rng.integers(4, 41))
                    duration_days = max(1, hours // 4)
                    end_date = start_date + pd.Timedelta(days=duration_days)

                    # Status: Completed / In Progress / Dropped
                    # Higher ability -> higher completion rate
                    p_complete = min(0.95, 0.70 + ability * 0.08)
                    p_dropped = max(0.02, 0.10 - ability * 0.03)
                    p_progress = 1.0 - p_complete - p_dropped
                    if p_progress < 0:
                        p_progress = 0.0
                        p_complete = 1.0 - p_dropped

                    status = str(self.rng.choice(
                        ["Completed", "In Progress", "Dropped"],
                        p=[p_complete, p_progress, p_dropped],
                    ))

                    # Score: only for completed trainings
                    score: float | None = None
                    if status == "Completed":
                        raw_score = 75.0 + ability * 5.0 + self.rng.normal(0.0, 8.0)
                        score = round(float(np.clip(raw_score, 50.0, 100.0)), 1)

                    # Cost: varies by category
                    base_costs = {
                        "Technical": 800, "Leadership": 1200, "Communication": 600,
                        "Compliance": 300, "Domain": 900,
                    }
                    cost = round(
                        base_costs[category] * (1.0 + self.rng.normal(0.0, 0.20)) * (hours / 20.0),
                        2,
                    )
                    cost = max(50.0, cost)

                    tr_counter += 1
                    rows.append({
                        "training_id": f"TR-{tr_counter:06d}",
                        "employee_id": emp_id,
                        "training_name": training_name,
                        "training_category": category,
                        "start_date": start_date,
                        "end_date": end_date,
                        "hours": hours,
                        "status": status,
                        "score": score,
                        "cost": cost,
                    })

        self._training_records = pd.DataFrame(rows)
        logger.info("Training records table: {} rows, {} columns", *self._training_records.shape)
        return self._training_records

    def generate_promotions(
        self,
        employees: pd.DataFrame,
        evaluations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate the promotions table.

        Roughly 15% of eligible employees per year receive a promotion, with
        higher-performing employees more likely to be promoted.

        Parameters
        ----------
        employees : pd.DataFrame
            The employees table.
        evaluations : pd.DataFrame
            The evaluations table (used to compute average performance).
        """
        logger.info("Generating promotions table ...")
        rows: list[dict] = []
        promo_counter = 0

        # Compute average performance per employee
        avg_perf = evaluations.groupby("employee_id")["performance_score"].mean()

        emp_indexed = employees.set_index("employee_id")

        # Iterate years 2021-2024; ~15% promotion rate per year
        for year in range(2021, 2025):
            for emp_id in employees["employee_id"].values:
                emp_row = emp_indexed.loc[emp_id]

                # Skip inactive
                if not emp_row["is_active"]:
                    continue

                # Must have been hired before this year
                hire_date = pd.Timestamp(emp_row["hire_date"])
                if hire_date.year >= year:
                    continue

                # Current level
                current_level = int(emp_row["job_level"])
                if current_level >= 6:
                    continue

                # Get performance (if available)
                perf = avg_perf.get(emp_id, 3.0)

                # Promotion probability: base ~15%/year adjusted by performance
                # perf 5.0 -> ~25%, perf 3.0 -> ~12%, perf 1.0 -> ~2%
                base_prob = 0.03  # per-year base (will be ~15% across the pool)
                perf_factor = max(0.1, (perf - 1.0) / 4.0)  # 0.0 to 1.0
                prob = base_prob + perf_factor * 0.08

                if self.rng.random() >= prob:
                    continue

                # Already promoted this function call? Track via level mutation
                new_level = current_level + 1
                dept = emp_row["department"]
                prev_title = DEPARTMENTS[dept]["titles"][current_level]
                new_title = DEPARTMENTS[dept]["titles"][new_level]

                # Salary increase: 5-25%, higher for bigger level jumps / better perf
                base_raise = 8.0 + perf_factor * 10.0
                raise_noise = self.rng.normal(0.0, 2.5)
                salary_increase_pct = round(float(np.clip(base_raise + raise_noise, 5.0, 25.0)), 2)

                # Performance at promotion: 3.5-5.0 (only good performers get promoted)
                perf_at_promo = round(float(np.clip(perf + self.rng.normal(0.2, 0.3), 3.5, 5.0)), 2)

                # Promotion date within the year
                promo_month = int(self.rng.choice([3, 6, 9, 12]))  # quarterly cycles
                promo_day = int(self.rng.integers(1, 29))
                promo_date = pd.Timestamp(year=year, month=promo_month, day=promo_day)

                promo_counter += 1
                rows.append({
                    "promotion_id": f"PROM-{promo_counter:04d}",
                    "employee_id": emp_id,
                    "promotion_date": promo_date,
                    "previous_level": current_level,
                    "new_level": new_level,
                    "previous_title": prev_title,
                    "new_title": new_title,
                    "salary_increase_pct": salary_increase_pct,
                    "performance_at_promotion": perf_at_promo,
                })

                # Update the index so same employee won't be re-promoted at same level
                emp_indexed.at[emp_id, "job_level"] = new_level

        self._promotions = pd.DataFrame(rows)
        logger.info("Promotions table: {} rows, {} columns", *self._promotions.shape)
        return self._promotions

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all four tables in dependency order.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``employees``, ``evaluations``, ``training_records``, ``promotions``.
        """
        logger.info(
            "Starting full synthetic data generation (n_employees={}, periods={})",
            self.n_employees,
            len(self.evaluation_periods),
        )

        employees = self.generate_employees()
        evaluations = self.generate_evaluations(employees)
        training_records = self.generate_training(employees)
        promotions = self.generate_promotions(employees, evaluations)

        logger.success(
            "Generation complete -- {} employees, {} evaluations, "
            "{} training records, {} promotions",
            len(employees),
            len(evaluations),
            len(training_records),
            len(promotions),
        )

        return {
            "employees": employees,
            "evaluations": evaluations,
            "training_records": training_records,
            "promotions": promotions,
        }

    def save_to_csv(self, output_dir: Path) -> None:
        """Save all generated tables to CSV files.

        If tables have not been generated yet, calls :meth:`generate_all` first.

        Parameters
        ----------
        output_dir : Path
            Directory where CSV files will be written. Created if missing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._employees is None:
            self.generate_all()

        tables = {
            "employees": self._employees,
            "evaluations": self._evaluations,
            "training_records": self._training_records,
            "promotions": self._promotions,
        }

        for name, df in tables.items():
            if df is None:
                logger.warning("Table '{}' is None, skipping.", name)
                continue
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("Saved {:>6} rows -> {}", len(df), path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for generating synthetic HR data."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic employee performance data.",
    )
    parser.add_argument(
        "-n", "--n-employees",
        type=int, default=1500,
        help="Number of employees to generate (default: 1500)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str, default="data/raw",
        help="Output directory for CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    gen = PerformanceDataGenerator(
        n_employees=args.n_employees,
        random_seed=args.seed,
    )
    gen.generate_all()
    gen.save_to_csv(Path(args.output_dir))


if __name__ == "__main__":
    main()
