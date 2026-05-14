
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
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

# ── Load model and config ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_model.pkl"))

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)

features          = config["features"]
APPROVE_THRESHOLD = config["approve_threshold"]
DECLINE_THRESHOLD = config["decline_threshold"]

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

# ── Input form ────────────────────────────────────────────────
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

# ── Auto-calculate fields ─────────────────────────────────────
if income > 0 and term_length > 0:
    install_to_inc = (loan_amount / term_length) / income * 100
else:
    install_to_inc = 0.0

now                 = datetime.datetime.now()
origination_date    = now - relativedelta(months=int(term_length))
origination_year    = origination_date.year
origination_quarter = (origination_date.month - 1) // 3 + 1

st.markdown(
    f"**Calculated instalment-to-income ratio: "
    f"{install_to_inc:.2f}%**")
st.divider()

# ── Build input dataframe ─────────────────────────────────────
input_data = pd.DataFrame([{
    "schufa":               schufa,
    "income":               income,
    "term_length":          term_length,
    "install_to_inc":       install_to_inc,
    "occup":                occupation,
    "marital":              marital,
    "loan_amount":          loan_amount,
    "num_applic":           num_applic,
    "origination_year":     origination_year,
    "origination_quarter":  origination_quarter
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

    feature_labels = {
        "schufa":               "SCHUFA Credit Score",
        "income":               "Monthly Income",
        "term_length":          "Loan Term Length",
        "install_to_inc":       "Repayment Burden",
        "occup":                "Employment Type",
        "marital":              "Marital Status",
        "loan_amount":          "Loan Amount",
        "num_applic":           "Number of Applicants",
        "origination_year":     "Loan Origination Year",
        "origination_quarter":  "Origination Quarter"
    }

    shap_df = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in features],
        "Impact":  shap_values[0]
    }).sort_values("Impact")

    total = shap_df["Impact"].abs().sum()
    shap_df["Weight"] = (
        shap_df["Impact"].abs() / total * 100
    ).round(1)

    approve_factors = shap_df[shap_df["Impact"] < 0].sort_values("Impact")
    decline_factors = shap_df[shap_df["Impact"] > 0].sort_values(
        "Impact", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        if decision == "APPROVE":
            st.success("### ✅ Reasons for Approval")
        else:
            st.success("### ✅ Factors in Applicant's Favour")

        for _, row in approve_factors.iterrows():
            weight = row["Weight"]
            if weight > 20:
                strength = "🟢 Strong positive"
            elif weight > 10:
                strength = "🟡 Moderate positive"
            else:
                strength = "⚪ Minor positive"
            st.markdown(
                f"**{row['Feature']}**  \n"
                f"{strength} — contributes "
                f"**{weight:.1f}%** of decision weight"
            )
            st.progress(min(weight / 40, 1.0))

    with col2:
        if decision == "DECLINE":
            st.error("### ❌ Reasons for Decline")
        else:
            st.warning("### ⚠️ Areas of Concern")

        for _, row in decline_factors.iterrows():
            weight = row["Weight"]
            if weight > 20:
                strength = "🔴 High concern"
            elif weight > 10:
                strength = "🟠 Moderate concern"
            else:
                strength = "🟡 Minor concern"
            st.markdown(
                f"**{row['Feature']}**  \n"
                f"{strength} — contributes "
                f"**{weight:.1f}%** of decision weight"
            )
            st.progress(min(weight / 40, 1.0))

    st.divider()

    top_approve = (approve_factors.iloc[0]
                   if len(approve_factors) > 0 else None)
    top_decline = (decline_factors.iloc[0]
                   if len(decline_factors) > 0 else None)

    if decision == "APPROVE":
        st.success(
            f"**Decision Summary:** This application was automatically "
            f"approved. The strongest positive factor was "
            f"**{top_approve['Feature']}** which accounted for "
            f"{top_approve['Weight']:.1f}% of the model's decision. "
            f"The predicted probability of default is "
            f"**{probability:.1%}** — below the approval "
            f"threshold of {APPROVE_THRESHOLD:.0%}."
        )
    elif decision == "REFER":
        st.warning(
            f"**Decision Summary:** This application has been referred "
            f"for manual underwriter review. The model identified mixed "
            f"signals — positive factors include "
            f"**{top_approve['Feature'] if top_approve is not None else 'N/A'}** "
            f"but concerns around "
            f"**{top_decline['Feature'] if top_decline is not None else 'N/A'}** "
            f"mean the decision requires human judgement. "
            f"Predicted default probability: **{probability:.1%}**."
        )
    else:
        st.error(
            f"**Decision Summary:** This application was automatically "
            f"declined. The primary concern was "
            f"**{top_decline['Feature']}** which accounted for "
            f"{top_decline['Weight']:.1f}% of the model's decision. "
            f"The predicted probability of default is "
            f"**{probability:.1%}** — above the decline threshold "
            f"of {DECLINE_THRESHOLD:.0%}."
        )

    st.divider()

    st.subheader("Applicant Summary")
    summary = pd.DataFrame({
        "Field": [
            "Monthly Income",
            "Loan Amount",
            "Term Length",
            "SCHUFA Credit Score",
            "Instalment to Income",
            "Number of Applicants",
            "Occupation",
            "Marital Status",
            "Origination Year",
            "Origination Quarter"
        ],
        "Value": [
            f"€{income:,} per month",
            f"€{loan_amount:,}",
            f"{term_length} months",
            f"{schufa_input}",
            f"{install_to_inc:.2f}%",
            num_applic,
            occupation,
            marital,
            origination_year,
            f"Q{origination_quarter}"
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
