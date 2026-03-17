"""Streamlit dashboard for Employee Performance Analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_API_BASE = "http://localhost:8000/api/v1"


# ------------------------------------------------------------------
# Model / API helpers
# ------------------------------------------------------------------


@st.cache_resource
def _load_model() -> Any | None:
    """Load the predictor model from disk (fallback when API is unavailable)."""
    model_path = _MODELS_DIR / "predictor.joblib"
    if not model_path.exists():
        return None
    try:
        import joblib

        artefact = joblib.load(model_path)
        return artefact["model"]
    except Exception:
        return None


@st.cache_resource
def _load_nine_box() -> Any:
    from src.models.nine_box import NineBoxClassifier

    return NineBoxClassifier()


@st.cache_resource
def _load_recommender() -> Any:
    from src.models.recommender import DevelopmentRecommender

    return DevelopmentRecommender()


@st.cache_resource
def _load_shap_analyzer() -> Any | None:
    model = _load_model()
    if model is None:
        return None
    try:
        from src.explainability.shap_analyzer import ShapAnalyzer

        return ShapAnalyzer(model, model_type="tree")
    except Exception:
        return None


def _call_api(endpoint: str, method: str = "GET", json_body: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Attempt an API call; return None on failure."""
    try:
        import httpx

        with httpx.Client(timeout=10) as client:
            if method == "POST":
                resp = client.post(f"{_API_BASE}{endpoint}", json=json_body)
            else:
                resp = client.get(f"{_API_BASE}{endpoint}")
            if resp.status_code == 200:
                return resp.json()  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


def _predict_local(features: dict[str, Any]) -> float | None:
    """Run prediction using the locally loaded model."""
    from src.models.predictor import PerformancePredictor

    model = _load_model()
    if model is None:
        return None
    row = {col: features.get(col, 0.0) for col in PerformancePredictor.FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=PerformancePredictor.FEATURE_COLUMNS)
    return float(model.predict(df)[0])


def _get_shap_drivers(features: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    """Return top SHAP drivers using local analyser."""
    from src.models.predictor import PerformancePredictor

    analyzer = _load_shap_analyzer()
    if analyzer is None:
        return []
    row = {col: features.get(col, 0.0) for col in PerformancePredictor.FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=PerformancePredictor.FEATURE_COLUMNS)
    try:
        return analyzer.get_top_drivers(df, index=0, top_n=top_n)  # type: ignore[no-any-return]
    except Exception:
        return []


# ------------------------------------------------------------------
# Plotly chart builders
# ------------------------------------------------------------------


def _build_nine_box_grid(perf: float, pot: float) -> go.Figure:
    """Create a 9-Box Grid scatter with quadrant boundaries."""
    fig = go.Figure()

    box_labels = {
        (0, 2): "Under Performer",
        (1, 2): "Development Needed",
        (2, 2): "Inconsistent Player",
        (0, 1): "Average Performer",
        (1, 1): "Core Player",
        (2, 1): "High Potential",
        (0, 0): "Solid Performer",
        (1, 0): "High Performer",
        (2, 0): "Star",
    }

    colours = {
        (0, 2): "#fee2e2",
        (1, 2): "#fef3c7",
        (2, 2): "#fef3c7",
        (0, 1): "#fef3c7",
        (1, 1): "#dbeafe",
        (2, 1): "#dbeafe",
        (0, 0): "#d1fae5",
        (1, 0): "#d1fae5",
        (2, 0): "#bbf7d0",
    }

    for (col, row_idx), label in box_labels.items():
        x0 = col * (5 / 3)
        x1 = (col + 1) * (5 / 3)
        y0 = (2 - row_idx) * (5 / 3)
        y1 = (3 - row_idx) * (5 / 3)
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=colours[(col, row_idx)],
            line=dict(color="#94a3b8", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=label,
            showarrow=False,
            font=dict(size=10, color="#475569"),
        )

    fig.add_trace(
        go.Scatter(
            x=[perf],
            y=[pot],
            mode="markers+text",
            marker=dict(size=16, color="#2563eb", symbol="diamond"),
            text=["Employee"],
            textposition="top center",
            name="Current Employee",
        )
    )

    fig.update_layout(
        title="9-Box Talent Grid",
        xaxis=dict(title="Performance", range=[0, 5], dtick=5 / 3),
        yaxis=dict(title="Potential", range=[0, 5], dtick=5 / 3),
        width=550,
        height=500,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def _build_gauge(score: float) -> go.Figure:
    """Create a performance prediction gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title=dict(text="Predicted Performance Score"),
            delta=dict(reference=3.0, increasing=dict(color="#16a34a"), decreasing=dict(color="#dc2626")),
            gauge=dict(
                axis=dict(range=[0, 5], tickwidth=1),
                bar=dict(color="#2563eb"),
                steps=[
                    dict(range=[0, 2.5], color="#fee2e2"),
                    dict(range=[2.5, 3.5], color="#fef3c7"),
                    dict(range=[3.5, 5], color="#d1fae5"),
                ],
                threshold=dict(line=dict(color="#dc2626", width=3), thickness=0.8, value=score),
            ),
        )
    )
    fig.update_layout(width=450, height=350, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def _build_shap_waterfall(drivers: list[dict[str, Any]]) -> go.Figure:
    """Create a horizontal bar chart mimicking a SHAP waterfall."""
    if not drivers:
        fig = go.Figure()
        fig.add_annotation(text="SHAP analysis unavailable", showarrow=False, font=dict(size=14))
        fig.update_layout(width=550, height=350)
        return fig

    features = [d["feature"] for d in reversed(drivers)]
    impacts = [d["shap_value"] for d in reversed(drivers)]
    colours = ["#16a34a" if v >= 0 else "#dc2626" for v in impacts]

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.3f}" for v in impacts],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top Feature Drivers (SHAP Values)",
        xaxis_title="SHAP Impact",
        yaxis_title="",
        width=550,
        height=max(300, len(drivers) * 35 + 100),
        margin=dict(l=160, r=40, t=50, b=40),
    )
    return fig


def _build_radar(profile: dict[str, float]) -> go.Figure:
    """Build a radar chart for competency profile comparison."""
    categories = list(profile.keys())
    values = list(profile.values())
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="#2563eb"),
            name="Employee",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="Competency Profile",
        width=500,
        height=420,
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


# ------------------------------------------------------------------
# Main dashboard
# ------------------------------------------------------------------


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Employee Performance Analytics",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Employee Performance Analytics")

    # ---- Sidebar inputs ----
    st.sidebar.header("Employee Attributes")

    age = st.sidebar.slider("Age", 18, 65, 35)
    department = st.sidebar.selectbox(
        "Department",
        ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Product", "Support"],
    )
    job_level = st.sidebar.slider("Job Level", 1, 10, 4)
    tenure_years = st.sidebar.slider("Tenure (years)", 0.0, 30.0, 5.0, step=0.5)
    avg_past_performance = st.sidebar.slider("Avg Past Performance", 0.0, 5.0, 3.2, step=0.1)
    performance_trend = st.sidebar.slider("Performance Trend", -0.5, 0.5, 0.05, step=0.01)
    total_training_hours = st.sidebar.slider("Training Hours", 0, 500, 80)
    training_completion_rate = st.sidebar.slider("Training Completion Rate", 0.0, 1.0, 0.85, step=0.05)
    avg_training_score = st.sidebar.slider("Avg Training Score", 0.0, 100.0, 72.0, step=1.0)
    overtime_hours = st.sidebar.slider("Monthly Overtime Hours", 0.0, 60.0, 8.0, step=1.0)
    goals_met_avg = st.sidebar.slider("Goals Met (avg)", 0.0, 1.0, 0.75, step=0.05)
    leadership_score = st.sidebar.slider("Leadership Score", 0.0, 5.0, 3.0, step=0.1)
    communication_score = st.sidebar.slider("Communication Score", 0.0, 5.0, 3.2, step=0.1)
    innovation_score = st.sidebar.slider("Innovation Score", 0.0, 5.0, 2.8, step=0.1)
    satisfaction_score = st.sidebar.slider("Satisfaction Score", 0.0, 5.0, 3.5, step=0.1)
    potential_score = st.sidebar.slider("Potential Score", 0.0, 5.0, 3.0, step=0.1)

    features: dict[str, Any] = {
        "avg_past_performance": avg_past_performance,
        "performance_trend": performance_trend,
        "total_training_hours": float(total_training_hours),
        "training_completion_rate": training_completion_rate,
        "avg_training_score": avg_training_score,
        "tenure_years": tenure_years,
        "job_level": job_level,
        "department_encoded": hash(department) % 20,
        "overtime_avg": overtime_hours,
        "goals_met_avg": goals_met_avg,
        "leadership_score": leadership_score,
        "communication_score": communication_score,
        "innovation_score": innovation_score,
    }

    # ---- Try API first, fall back to local model ----
    api_payload = {
        "age": age,
        "department": department,
        "job_level": job_level,
        "tenure_years": tenure_years,
        "avg_past_performance": avg_past_performance,
        "performance_trend": performance_trend,
        "total_training_hours": float(total_training_hours),
        "training_completion_rate": training_completion_rate,
        "avg_training_score": avg_training_score,
        "overtime_hours": overtime_hours,
        "goals_met_avg": goals_met_avg,
        "leadership_score": leadership_score,
        "communication_score": communication_score,
        "innovation_score": innovation_score,
        "satisfaction_score": satisfaction_score,
        "potential_score": potential_score,
    }

    api_result = _call_api("/predict", method="POST", json_body=api_payload)

    if api_result is not None:
        predicted_score = api_result["predicted_score"]
        shap_drivers = [
            {
                "feature": d["feature"],
                "value": d["value"],
                "shap_value": d["impact"],
                "direction": d["direction"],
            }
            for d in api_result.get("top_drivers", [])
        ]
        nine_box_label = api_result.get("nine_box_label", "")
        recommendations = api_result.get("recommendations", [])
    else:
        predicted_score = _predict_local(features)
        shap_drivers = _get_shap_drivers(features, top_n=10)

        nine_box_classifier = _load_nine_box()
        if predicted_score is not None:
            nb_result = nine_box_classifier.classify(predicted_score, potential_score)
            nine_box_label = nb_result["box_label"]
        else:
            nine_box_label = "N/A"

        recommender = _load_recommender()
        profile_for_recs: dict[str, Any] = {
            "box_label": nine_box_label,
            "leadership_score": leadership_score,
            "communication_score": communication_score,
            "innovation_score": innovation_score,
            "training_completion_rate": training_completion_rate,
            "total_training_hours": float(total_training_hours),
            "performance_trend": performance_trend,
            "goals_met_avg": goals_met_avg,
        }
        recommendations = recommender.recommend(profile_for_recs)[:5]

    if predicted_score is None:
        st.warning("No model available. Load a trained model into the models/ directory to enable predictions.")
        predicted_score = avg_past_performance

    # ---- Layout ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            _build_nine_box_grid(predicted_score, potential_score),
            use_container_width=True,
        )

    with col_right:
        st.plotly_chart(
            _build_gauge(predicted_score),
            use_container_width=True,
        )

    st.divider()

    col_shap, col_radar = st.columns(2)

    with col_shap:
        st.plotly_chart(
            _build_shap_waterfall(shap_drivers),
            use_container_width=True,
        )

    with col_radar:
        competency_profile = {
            "Leadership": leadership_score,
            "Communication": communication_score,
            "Innovation": innovation_score,
            "Goal Achievement": goals_met_avg * 5,
            "Training": training_completion_rate * 5,
            "Satisfaction": satisfaction_score,
        }
        st.plotly_chart(
            _build_radar(competency_profile),
            use_container_width=True,
        )

    st.divider()

    # ---- Development Recommendations ----
    st.subheader("Development Recommendations")

    if recommendations:
        for rec in recommendations:
            priority = rec.get("priority", "Medium")
            colour_map = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}
            border_colour = colour_map.get(priority, "#6b7280")
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {border_colour};
                    padding: 12px 16px;
                    margin-bottom: 12px;
                    background: #f8fafc;
                    border-radius: 4px;
                ">
                    <strong style="color: {border_colour};">[{priority}]</strong>
                    <strong>{rec.get('area', '')}</strong><br/>
                    {rec.get('action', '')}<br/>
                    <em style="color: #64748b;">Expected impact: {rec.get('expected_impact', 'N/A')}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No specific development recommendations at this time.")


if __name__ == "__main__":
    main()
