import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_deployment_model.pkl"

pipeline = joblib.load(MODEL_PATH)


def test_model_prediction():

    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 840.0
    }])

    prediction = pipeline.predict(sample)

    assert prediction[0] in ["Yes", "No"]


def test_prediction_probability():

    sample = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 1,
        "tenure": 5,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95,
        "TotalCharges": 500
    }])

    probability = pipeline.predict_proba(sample)[0][1]

    assert 0 <= probability <= 1