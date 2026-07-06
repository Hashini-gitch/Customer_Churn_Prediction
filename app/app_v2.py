import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_deployment_model.pkl"

pipeline = joblib.load(MODEL_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer will leave the company.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

st.divider()

# -----------------------------
# Create DataFrame
# -----------------------------
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "tenure": tenure,
    "InternetService": internet,
    "Contract": contract,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn"):

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()

    st.subheader("📋 Prediction Result")

    if prediction == "Yes":
        st.error("⚠️ Customer is likely to CHURN")

    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )