import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Eligibility Dashboard",
    page_icon="??",
    layout="wide"
)

MODEL_PATH = "tuned_random_forest.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("?? Loan Eligibility Prediction Dashboard")
st.caption("Machine Learning • Explainable AI • Responsible AI")

# ---------------- Prediction ----------------

st.sidebar.header("Loan Application")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 720)
annual_income = st.sidebar.number_input("Annual Income", 0.0, 10000000.0, 75000.0)
loan_amount = st.sidebar.number_input("Loan Amount", 0.0, 10000000.0, 250000.0)
loan_term_months = st.sidebar.number_input("Loan Term (Months)", 1.0, 600.0, 240.0)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Employed", "Self-employed", "Unemployed"]
)

property_ownership = st.sidebar.selectbox(
    "Property Ownership",
    ["Own", "Rent", "Mortgage"]
)

debt_to_income_ratio = st.sidebar.number_input(
    "Debt-to-Income Ratio", 0.0, 2.0, 0.30
)

num_dependents = st.sidebar.number_input(
    "Number of Dependents", 0.0, 20.0, 2.0
)

monthly_income = st.sidebar.number_input(
    "Monthly Income", 0.0, 1000000.0, 6250.0
)

estimated_monthly_payment = st.sidebar.number_input(
    "Estimated Monthly Payment", 0.0, 1000000.0, 1800.0
)

total_dti = st.sidebar.number_input(
    "Total DTI", 0.0, 2.0, 0.45
)

input_data = pd.DataFrame([{
    "credit_score": credit_score,
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "employment_status": employment_status,
    "property_ownership": property_ownership,
    "debt_to_income_ratio": debt_to_income_ratio,
    "num_dependents": num_dependents,
    "monthly_income": monthly_income,
    "estimated_monthly_payment": estimated_monthly_payment,
    "total_dti": total_dti
}])

st.header("Loan Prediction")

if st.button("Predict Loan Eligibility", type="primary"):
    prediction = int(model.predict(input_data)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("### Eligible")
        else:
            st.error("### Not Eligible")

    with col2:
        if probability is not None:
            st.metric(
                "Model Eligibility Probability",
                f"{probability:.2%}"
            )

st.divider()

# ---------------- Model Metrics ----------------

st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Accuracy", "100%")

with col3:
    st.metric("Dataset", "10,000 records")

st.caption(
    "The reported accuracy is for the engineered experimental target used in this project."
)

st.divider()

# ---------------- SHAP ----------------

st.header("Explainable AI — SHAP")

st.write(
    "SHAP was used to identify which features contributed most strongly "
    "to the model predictions."
)

shap_col1, shap_col2 = st.columns(2)

with shap_col1:
    shap_importance = "experiment_5_outputs/shap_feature_importance.png"

    if os.path.exists(shap_importance):
        st.subheader("Feature Importance")
        st.image(shap_importance, use_container_width=True)
    else:
        st.info("SHAP feature importance plot not found.")

with shap_col2:
    shap_summary = "experiment_5_outputs/shap_summary.png"

    if os.path.exists(shap_summary):
        st.subheader("SHAP Summary")
        st.image(shap_summary, use_container_width=True)
    else:
        st.info("SHAP summary plot not found.")

st.divider()

# ---------------- LIME ----------------

st.header("Explainable AI — LIME")

st.write(
    "LIME provides a local explanation for an individual model prediction."
)

lime_file = "experiment_5_outputs/lime_explanation.html"

if os.path.exists(lime_file):
    st.success("LIME explanation generated successfully.")
    st.caption("Open the generated LIME HTML report from the experiment output folder.")
else:
    st.info("LIME explanation file not found.")

st.divider()

# ---------------- Fairness ----------------

st.header("Fairness Audit")

fairness_file = "experiment_5_outputs/fairness_audit_report.txt"

if os.path.exists(fairness_file):
    with open(fairness_file, "r", encoding="utf-8") as file:
        fairness_report = file.read()

    st.text_area(
        "Fairness Audit Report",
        fairness_report,
        height=250
    )
else:
    st.info("Fairness audit report not found.")

st.divider()

# ---------------- Responsible AI ----------------

st.header("Responsible AI Checklist")

responsible_ai = {
    "Fairness": "Evaluate prediction disparities across relevant groups.",
    "Privacy": "Collect and process only information required for the intended task.",
    "Consent": "Users should understand how their submitted information is used.",
    "Transparency": "Model predictions should not be treated as guaranteed loan decisions.",
    "Human Oversight": "Automated predictions should support appropriate human review.",
    "Monitoring": "Monitor model performance, fairness, and data drift after deployment."
}

for category, description in responsible_ai.items():
    st.markdown(f"**{category}:** {description}")

st.warning(
    "Important: The loan eligibility target in this project was engineered "
    "from financial variables. Therefore, the model and fairness results "
    "represent this experimental dataset and target-generation rule, "
    "not evidence of a real-world lending system."
)

st.divider()

st.caption("Loan Eligibility Prediction — Final ML Portfolio Project")
