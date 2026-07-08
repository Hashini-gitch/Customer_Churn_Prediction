import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def get_risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def explain_customer(data):

    reasons = []

    if data["Contract"].iloc[0] == "Month-to-month":
        reasons.append("📌 Month-to-month contracts have higher churn risk.")

    if data["InternetService"].iloc[0] == "Fiber optic":
        reasons.append("📌 Fiber optic customers historically churn more often.")

    if data["PaymentMethod"].iloc[0] == "Electronic check":
        reasons.append("📌 Electronic check users have higher churn rates.")

    if data["MonthlyCharges"].iloc[0] > 80:
        reasons.append("📌 High monthly charges increase churn risk.")

    if data["tenure"].iloc[0] < 12:
        reasons.append("📌 New customers are more likely to churn.")

    return reasons


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_deployment_model.pkl"

HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

HISTORY_PATH = HISTORY_DIR / "prediction_history.csv"

pipeline = joblib.load(MODEL_PATH)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("📘 About")

st.sidebar.info(
"""
### Customer Churn Prediction System

This AI application predicts whether a telecom customer is likely to leave the company.

### Model
- Logistic Regression

### Features
- Gender
- Senior Citizen
- Tenure
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges

### Built With
- Python
- Scikit-learn
- Streamlit
"""
)

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

st.divider()

# -------------------------------------------------
# Input Form
# -------------------------------------------------

left, right = st.columns(2)

with left:

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

with right:

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

# -------------------------------------------------
# Input DataFrame
# -------------------------------------------------

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

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔍 Predict Churn"):

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0][1]

    risk_label = get_risk_level(probability)

    history = pd.DataFrame([{

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": prediction,
        "Probability": round(probability,4),
        "Risk": risk_label

    }])

    history.to_csv(

        HISTORY_PATH,
        mode="a",
        header=not os.path.exists(HISTORY_PATH),
        index=False

    )

    st.divider()

    # -------------------------
    # Prediction Result
    # -------------------------

    st.subheader("📋 Prediction Result")

    if prediction == "Yes":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    # -------------------------
    # Risk Analysis
    # -------------------------

    st.subheader("📊 Risk Analysis")

    st.markdown(f"## {risk_label}")

    st.progress(float(probability))

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    # -------------------------
    # Business Recommendation
    # -------------------------

    st.subheader("💡 Business Recommendation")

    if risk_label == "High Risk":

        st.error(
            "Offer retention discounts, loyalty rewards or customer support intervention."
        )

    elif risk_label == "Medium Risk":

        st.warning(
            "Monitor this customer and consider personalized offers."
        )

    else:

        st.success(
            "Customer appears stable. No immediate action is required."
        )

    # -------------------------
    # Explanation
    # -------------------------

    st.subheader("🔍 Why this prediction?")

    reasons = explain_customer(input_df)

    if len(reasons) == 0:

        st.success("No major churn risk factors detected.")

    else:

        for reason in reasons:
            st.write(reason)

# -------------------------------------------------
# Prediction History
# -------------------------------------------------

st.divider()

st.subheader("📜 Prediction History")

if os.path.exists(HISTORY_PATH):

    history_df = pd.read_csv(HISTORY_PATH)

    if not history_df.empty:

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

    else:

        st.info("No predictions made yet.")

else:

    st.info("Prediction history file not found.")

# -------------------------------------------------
# Analytics Dashboard
# -------------------------------------------------

st.divider()

st.subheader("📊 Analytics Dashboard")

if os.path.exists(HISTORY_PATH):

    history_df = pd.read_csv(HISTORY_PATH)

    if not history_df.empty:

        total_predictions = len(history_df)

        high_risk = (
            history_df["Risk"] == "High Risk"
        ).sum()

        avg_probability = history_df["Probability"].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            total_predictions
        )

        col2.metric(
            "High Risk Customers",
            high_risk
        )

        col3.metric(
            "Average Probability",
            f"{avg_probability*100:.1f}%"
        )

        # -------------------------
        # Pie Chart
        # -------------------------

        prediction_counts = history_df["Prediction"].value_counts()

        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(

            prediction_counts,
            labels=prediction_counts.index,
            autopct="%1.1f%%"

        )

        ax.set_title("Prediction Distribution")

        st.pyplot(fig)

        # -------------------------
        # Bar Chart
        # -------------------------

        risk_counts = history_df["Risk"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(6,4))

        ax2.bar(

            risk_counts.index,
            risk_counts.values

        )

        ax2.set_title("Risk Distribution")

        ax2.set_ylabel("Customers")

        st.pyplot(fig2)

    else:

        st.info("No analytics available yet.")

else:

    st.info("Prediction history file not found.")

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Developed by Hashini Avishka | Data Science Portfolio Project"
)