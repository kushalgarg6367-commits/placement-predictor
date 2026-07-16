"""
app.py
------
Streamlit app: enter a student profile, get a placement probability
prediction plus a SHAP explanation of *why* the model predicted that.

Run locally:
    streamlit run app.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.preprocess import BRANCH_ORDER, FEATURE_COLUMNS, encode
from src.scoring import compute_coding_score, compute_communication_score

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("models/placement_model.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return model, metrics


@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


model, metrics = load_model()

st.title("🎓 Campus Placement Predictor")
st.caption(
    "Enter a student profile below to estimate placement probability. "
    "Trained on a synthetic dataset modeled on real placement drivers "
    f"(model: {metrics['backend']}, test accuracy: {metrics['model_accuracy']:.0%})."
)

with st.form("student_form"):
    col1, col2 = st.columns(2)
    with col1:
        branch = st.selectbox("Branch", BRANCH_ORDER)
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.05)
        internships = st.selectbox("Internships completed", [0, 1, 2, 3])
        projects = st.selectbox("Projects completed", [0, 1, 2, 3, 4])
        backlogs = st.selectbox("Current backlogs", [0, 1, 2, 3])
        certifications = st.number_input("Certifications completed", 0, 10, 1)
        extracurricular = st.selectbox("Active in extracurriculars?", ["No", "Yes"])

    with col2:
        st.markdown("**DSA practice** (LeetCode + Codeforces combined)")
        easy_solved = st.number_input("Easy problems solved", 0, 500, 30, step=5)
        medium_solved = st.number_input("Medium problems solved", 0, 400, 15, step=5)
        hard_solved = st.number_input("Hard problems solved", 0, 200, 2, step=1)

        st.markdown("**Communication self-assessment**")
        group_comfort = st.select_slider(
            "Comfort speaking in group discussions / mock interviews",
            options=[1, 2, 3, 4, 5], value=3,
            format_func=lambda x: {1: "Very uncomfortable", 2: "Uncomfortable",
                                    3: "Neutral", 4: "Comfortable",
                                    5: "Very comfortable"}[x],
        )
        past_feedback = st.selectbox(
            "Feedback received in past interviews / presentations",
            ["No feedback yet", "Mostly negative", "Mixed", "Mostly positive"],
        )
        fluency = st.select_slider(
            "Fluency explaining technical concepts out loud",
            options=[1, 2, 3, 4, 5], value=3,
            format_func=lambda x: {1: "Struggle a lot", 2: "Somewhat struggle",
                                    3: "Neutral", 4: "Fairly fluent",
                                    5: "Very fluent"}[x],
        )

    submitted = st.form_submit_button("Predict placement chance")

if submitted:
    coding_score = compute_coding_score(easy_solved, medium_solved, hard_solved)
    communication_score = compute_communication_score(group_comfort, past_feedback, fluency)

    raw = pd.DataFrame([{
        "branch": branch,
        "cgpa": cgpa,
        "internships": internships,
        "projects_completed": projects,
        "backlogs": backlogs,
        "coding_score": coding_score,
        "communication_score": communication_score,
        "certifications": certifications,
        "extracurricular": 1 if extracurricular == "Yes" else 0,
    }])

    X = encode(raw)[FEATURE_COLUMNS]
    proba = float(model.predict_proba(X)[0, 1])

    st.divider()
    st.subheader("Prediction")

    c1, c2 = st.columns(2)
    c1.metric("Computed coding score", f"{coding_score:.0f} / 100")
    c2.metric("Computed communication score", f"{communication_score:.0f} / 100")

    st.metric("Estimated placement probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= 0.65:
        st.success("Strong profile for placements based on current inputs.")
    elif proba >= 0.4:
        st.warning("Borderline profile — a few improvements could meaningfully help.")
    else:
        st.error("Profile suggests placement risk — see the biggest factors below.")

    st.subheader("Why this prediction? (SHAP explanation)")
    try:
        explainer = load_explainer(model)
        shap_values = explainer.shap_values(X)

        # SHAP's return shape varies by model/version:
        # - list of arrays (one per class) for some sklearn classifiers
        # - (n_samples, n_features) for binary XGBoost
        # - (n_samples, n_features, n_classes) in newer SHAP releases
        if isinstance(shap_values, list):
            row_shap = shap_values[1][0]          # positive class, first (only) row
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            row_shap = shap_values[0, :, 1]        # first row, positive class
        else:
            row_shap = shap_values[0]              # (n_samples, n_features)

        contrib = pd.Series(row_shap, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=True)
        contrib = contrib[contrib.abs() > 1e-6].tail(8)  # top 8 drivers for this student

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in contrib.values]
        ax.barh(contrib.index, contrib.values, color=colors)
        ax.set_xlabel("Impact on placement probability (SHAP value)")
        ax.set_title("Top factors driving this prediction")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "🟢 Green bars push the prediction toward *placed*. "
            "🔴 Red bars push it toward *not placed*. Longer bar = bigger impact."
        )
    except ImportError:
        st.info("Install `shap` (`pip install shap`) to see a per-student explanation here.")

st.divider()
with st.expander("About this project"):
    st.write(
        "This model is trained on a synthetically generated dataset built from "
        "realistic placement drivers (CGPA, internships, projects, coding/communication "
        "skills, backlogs). It's meant as a learning/demo tool, not a real placement "
        "prediction service — the underlying data is not from actual MNIT placement records."
    )
    st.markdown(
        "**How coding/communication scores are computed:**\n"
        "- Coding score: a weighted, diminishing-returns formula over easy/medium/hard "
        "DSA problems solved on LeetCode + Codeforces (see `src/scoring.py`)\n"
        "- Communication score: a weighted blend of group-discussion comfort, past "
        "interview/presentation feedback, and self-rated fluency"
    )
    st.json(metrics)
