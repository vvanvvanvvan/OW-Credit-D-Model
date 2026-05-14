
import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load model and config ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_model.pkl"))

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)

features          = config["features"]
APPROVE_THRESHOLD = config["approve_threshold"]
DECLINE_THRESHOLD = config["decline_threshold"]

# ── Auto-calculate time features ──────────────────────────────
now         = datetime.datetime.now()
obs_quarter = (now.month - 1) // 3 + 1

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Decisioning Model",
    page_icon="🏦",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("🏦 Credit Decisioning Model")
st.markdown(
    "Enter applicant details below to receive an instant credit decision.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Financial Details")
    income      = st.number_input(
        "Monthly Income (€)",
        min_value=0,
        value=2500,
        step=100)
    loan_amount = st.number_input(
        "Loan Amount (€)",
        min_value=0,
        value=15000,
        step=500)
    term_length = st.number_input(
        "Term Length (months)",
        min_value=1,
        value=60,
        step=6)

with col2:
    st.subheader("Credit Profile")
    schufa_input = st.number_input(
        "SCHUFA Credit Score",
        min_value=100,
        max_value=999,
        value=587,
        step=1,
        help="Enter standard SCHUFA score (100-999). "
             "Higher scores indicate better creditworthiness.")

    schufa = schufa_input * 15

    num_applic = st.selectbox(
        "Number of Applicants", [1, 2])

with col3:
    st.subheader("Personal Details")
    occupation = st.selectbox(
        "Occupation",
        ["Employee", "Student", "Worker"])
    marital    = st.selectbox(
        "Marital Status",
        ["Married", "Living together", "Single",
         "Separated", "Divorced"])

st.divider()

# ── Auto-calculate instalment to income ──────────────────────
if income > 0 and term_length > 0:
    install_to_inc = (loan_amount / term_length) / income * 100
else:
    install_to_inc = 0.0

# ── Auto-calculate origination year ──────────────────────────
from dateutil.relativedelta import relativedelta
obs_date         = datetime.datetime.now()
origination_date = obs_date - relativedelta(months=int(term_length))
origination_year = origination_date.year

st.markdown(
    f"**Calculated instalment-to-income ratio: "
    f"{install_to_inc:.2f}%**")
st.divider()

# ── Build input dataframe ─────────────────────────────────────
input_data = pd.DataFrame([{
    "schufa":            schufa,
    "income":            income,
    "term_length":       term_length,
    "install_to_inc":    install_to_inc,
    "occup":             occupation,
    "marital":           marital,
    "loan_amount":       loan_amount,
    "num_applic":        num_applic,
    "origination_year":  origination_year,
    "obs_quarter":       obs_quarter
}])

input_data["occup"]   = input_data["occup"].astype("category")
input_data["marital"] = input_data["marital"].astype("category")

# ── Run decision ──────────────────────────────────────────────
if st.button("Run Credit Decision",
             type="primary",
             use_container_width=True):

    probability = model.predict_proba(input_data)[0][1]

    if probability < APPROVE_THRESHOLD:
        decision = "APPROVE"
        icon     = "✅"
        message  = "Low default risk — auto approved"
    elif probability < DECLINE_THRESHOLD:
        decision = "REFER"
        icon     = "⚠️"
        message  = "Moderate risk — refer for manual review"
    else:
        decision = "DECLINE"
        icon     = "❌"
        message  = "High default risk — auto declined"

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Decision", f"{icon} {decision}")
    with col2:
        st.metric("Default Probability", f"{probability:.1%}")
    with col3:
        st.metric("Risk Band",
                  "Low"    if decision == "APPROVE"
                  else "Medium" if decision == "REFER"
                  else "High")
    with col4:
        st.metric("Instalment Ratio", f"{install_to_inc:.2f}%")

    st.markdown(f"**{message}**")
    st.divider()

    st.subheader("Decision Thresholds")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"✅ **APPROVE**  \nBelow "
                f"{APPROVE_THRESHOLD:.0%} default probability")
    with col2:
        st.warning(f"⚠️ **REFER**  \nBetween "
                   f"{APPROVE_THRESHOLD:.0%} and "
                   f"{DECLINE_THRESHOLD:.0%}")
    with col3:
        st.error(f"❌ **DECLINE**  \nAbove "
                 f"{DECLINE_THRESHOLD:.0%} default probability")

    st.divider()

    st.subheader("Factors Impacting the Decision")
    st.markdown(
        "The factors below show what drove this credit decision "
        "in plain terms.")
    st.divider()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_data.iloc[0],
            feature_names = features
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.subheader("Key risk factors")
    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact":  shap_values[0]
    }).sort_values("Impact")

    top_risk    = shap_df.tail(3)
    top_protect = shap_df.head(3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factors increasing risk:**")
        for _, row in top_risk.iterrows():
            st.markdown(f"- {row['Feature']}: +{row['Impact']:.3f}")
    with col2:
        st.markdown("**Factors reducing risk:**")
        for _, row in top_protect.iterrows():
            st.markdown(f"- {row['Feature']}: {row['Impact']:.3f}")
