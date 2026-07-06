import streamlit as st
import joblib

model = joblib.load("models/customer_churn_model.pkl")

st.success("Model loaded successfully!")

st.title("Customer Churn Prediction System")
st.write("This AI application predicts whether a customer is likely to churn")
st.divider()
st.subheader("Customer Information")
st.write("Please enter the customer details below.")
gender = st.selectbox(
    "Gender", 
    ["Female", "Male"]
)
st.write("Selected Gender:", gender)
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)
st.write("Tenure:", tenure)
senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)
total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)
contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)
internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
st.divider()

st.subheader("Customer Summary")

st.write("Gender:", gender)
st.write("Senior Citizen:", senior_citizen)
st.write("Tenure:", tenure)
st.write("Monthly Charges:", monthly_charges)
st.write("Total Charges:", total_charges)
st.write("Contract:", contract)
st.write("Internet Service:", internet_service)
st.write("Payment Method:", payment_method)

st.divider()

st.subheader("Data Conversion")

if gender == "Male":
    gender_value = 1
else:
    gender_value = 0

    st.write("Encoded Gender:", gender_value)

    if senior_citizen == "Yes":
        senior_value = 1
    else:
        senior_value = 0

    st.write("Encoded Senior Citizen:", senior_value)

    contract_one_year = 0
    contract_two_year = 0

    if contract == "One year":
        contract_one_year = 1
    elif contract == "Two year":
        contract_two_year = 1

        st.write("Contract One Year:", contract_one_year)
        st.write("Contract Two Year:", contract_two_year)

        internet_fiber = 0
        internet_no = 0

        if internet_service == "Fiber optic":
            internet_fiber = 1
        elif internet_service == "No":
            internet_no = 1

        st.write("Fiber Optic:", internet_fiber)
        st.write("No Service:", internet_no)

        payment_electronic = 0
        payment_mailed = 0
        payment_bank = 0

        if payment_method == "Electronic check":
            payment_electronic = 1
        elif payment_method == "Mailed check":
            payment_mailed = 1
        elif payment_method == "Bank transfer (automatic)":
            payment_bank = 1

        st.write("Electronic Check:", payment_electronic)
        st.write("Mailed Check:", payment_mailed)
        st.write("Bank Transfer:", payment_bank)

        st.divider()

        if st.button("Predict Churn"):
           st.write("Model is making prediction...")

           input_data = [
                gender_value,
                tenure,
                senior_value,
                monthly_charges,
                total_charges,
                contract_one_year,
                contract_two_year,
                internet_fiber,
                internet_no,
                payment_electronic,
                payment_mailed,
                payment_bank
        ]

import numpy as np

input_array = np.array(input_data).reshape(1, -1)

prediction = model.predict(input_array)

if prediction[0] == 1:
    st.error("⚠️ Customer is likely to CHURN")
else:
    st.success("✅ Customer will STAY")

prob = model.predict_proba(input_array)[0][1]

st.write("Churn Probability:", prob)
