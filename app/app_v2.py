import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

from pathlib import Path
from datetime import datetime


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_deployment_model.pkl"

EXPLAINER_PATH = BASE_DIR / "models" / "shap_explainer.pkl"

HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

HISTORY_PATH = HISTORY_DIR / "prediction_history.csv"


# ==========================================================
# LOAD MODEL
# ==========================================================

pipeline = joblib.load(MODEL_PATH)

explainer = joblib.load(EXPLAINER_PATH)


# ==========================================================
# FUNCTIONS
# ==========================================================

def get_risk_level(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    else:
        return "High Risk"


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📘 About This Project")

st.sidebar.markdown("""
### Customer Churn Prediction

Predict whether a telecom customer will leave the company.

### Machine Learning Model

- Logistic Regression
- Scikit-Learn Pipeline

### Explainability

- SHAP Explainability

### Technologies

- Python
- Streamlit
- Pandas
- Matplotlib
- SHAP
- Scikit-learn
""")


# ==========================================================
# TITLE
# ==========================================================

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer is likely to churn using Machine Learning."
)

st.divider()


# ==========================================================
# INPUTS
# ==========================================================

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
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


with right:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
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


# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

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

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🔍 Predict Churn"):

    # -----------------------------
    # Model Prediction
    # -----------------------------

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0][1]

    risk_label = get_risk_level(probability)

    # -----------------------------
    # Save Prediction History
    # -----------------------------

    history = pd.DataFrame([{

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Prediction": prediction,

        "Probability": round(probability, 4),

        "Risk": risk_label

    }])

    history.to_csv(

        HISTORY_PATH,

        mode="a",

        header=not HISTORY_PATH.exists(),

        index=False

    )

    st.divider()

    # =====================================================
    # Prediction Result
    # =====================================================

    st.subheader("📋 Prediction Result")

    if prediction == "Yes":

        st.error("⚠️ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is likely to STAY")

    # =====================================================
    # Risk Analysis
    # =====================================================

    st.subheader("📊 Risk Analysis")

    st.metric(

        label="Churn Probability",

        value=f"{probability*100:.2f}%"

    )

    st.progress(float(probability))

    st.markdown(f"### Risk Level: **{risk_label}**")

    # =====================================================
    # Business Recommendation
    # =====================================================

    st.subheader("💡 Business Recommendation")

    if risk_label == "High Risk":

        st.error(
            """
Offer a retention discount.

Provide loyalty rewards.

Arrange a proactive customer support call.
"""
        )

    elif risk_label == "Medium Risk":

        st.warning(
            """
Monitor this customer carefully.

Recommend personalized offers.
"""
        )

    else:

        st.success(
            """
Customer appears stable.

No immediate action is required.
"""
        )

    # =====================================================
    # SHAP Explainability
    # =====================================================

    st.divider()

    st.subheader("🧠 AI Explainability (SHAP)")

    # Transform input

    transformed = pipeline.named_steps[
        "preprocessor"
    ].transform(input_df)

    # Calculate SHAP values

    shap_values = explainer(transformed)

    # Waterfall Plot

    fig = plt.figure(figsize=(10,5))

    shap.plots.waterfall(

        shap_values[0],

        max_display=8,

        show=False

    )

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # Feature Contribution Table
    # =====================================================

    st.subheader("📈 Top Feature Contributions")

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Contribution": shap_values.values[0]

    })

    importance_df["Absolute"] = (
        importance_df["Contribution"].abs()
    )

    importance_df = importance_df.sort_values(

        "Absolute",

        ascending=False

    )

    st.dataframe(

        importance_df[
            ["Feature", "Contribution"]
        ].head(10),

        width="stretch"

    )

    # =====================================================
    # Download Prediction
    # =====================================================

    st.subheader("📥 Download Prediction")

    result_df = pd.DataFrame([{

        "Prediction": prediction,

        "Probability": round(probability,4),

        "Risk": risk_label

    }])

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="⬇ Download Result as CSV",

        data=csv,

        file_name="prediction_result.csv",

        mime="text/csv"

    )

    # ==========================================================
# PREDICTION HISTORY
# ==========================================================

st.divider()

st.subheader("📜 Prediction History")

if HISTORY_PATH.exists():

    history_df = pd.read_csv(HISTORY_PATH)

    if not history_df.empty:

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

    else:
        st.info("No predictions have been made yet.")

else:
    st.info("Prediction history file not found.")

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

st.divider()

st.header("📊 Analytics Dashboard")

if HISTORY_PATH.exists():

    history_df = pd.read_csv(HISTORY_PATH)

    if not history_df.empty:

        total_predictions = len(history_df)

        churn_predictions = (
            history_df["Prediction"] == "Yes"
        ).sum()

        stay_predictions = (
            history_df["Prediction"] == "No"
        ).sum()

        high_risk = (
            history_df["Risk"] == "High Risk"
        ).sum()

        avg_probability = history_df[
            "Probability"
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Predictions",
            total_predictions
        )

        col2.metric(
            "High Risk",
            high_risk
        )

        col3.metric(
            "Likely Churn",
            churn_predictions
        )

        col4.metric(
            "Average Risk",
            f"{avg_probability*100:.1f}%"
        )

        st.divider()

        # ==========================================
        # Prediction Distribution
        # ==========================================

        left_chart, right_chart = st.columns(2)

        with left_chart:

            st.subheader("Prediction Distribution")

            prediction_counts = history_df[
                "Prediction"
            ].value_counts()

            fig1, ax1 = plt.subplots(figsize=(5,5))

            ax1.pie(
                prediction_counts.values,
                labels=prediction_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax1.axis("equal")

            st.pyplot(fig1)

            plt.close(fig1)

        # ==========================================
        # Risk Distribution
        # ==========================================

        with right_chart:

            st.subheader("Risk Distribution")

            risk_counts = history_df[
                "Risk"
            ].value_counts()

            fig2, ax2 = plt.subplots(figsize=(6,4))

            ax2.bar(
                risk_counts.index,
                risk_counts.values
            )

            ax2.set_ylabel("Customers")

            ax2.set_title("Risk Levels")

            st.pyplot(fig2)

            plt.close(fig2)

        st.divider()

        # ==========================================
        # Recent Predictions
        # ==========================================

        st.subheader("📋 Recent Predictions")

        st.dataframe(

            history_df.tail(20),

            use_container_width=True

        )

else:

    st.info(
        "Run a prediction first to generate analytics."
    )

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.divider()

with st.expander("ℹ️ Project Information"):

    st.markdown("""
### Customer Churn Prediction System

This project predicts whether a telecom customer is likely to leave the company.

### Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- ColumnTransformer
- One-Hot Encoding
- StandardScaler
- Logistic Regression

### Explainability

- SHAP Explainable AI (XAI)

### Deployment

- Streamlit

### Developer

**Hashini Avishka**

Data Science Undergraduate
""")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "© 2026 Hashini Avishka | Customer Churn Prediction System | Built with Streamlit & Scikit-learn"
)