import streamlit as st
import pandas as pd
import joblib


model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Prediction App")

st.write("Enter applicant details below:")


gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income")
co_income = st.number_input("Co-applicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")

credit_history = st.selectbox("Credit History", [1, 0])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])



gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

dependents = 3 if dependents == "3+" else int(dependents)

urban = 1 if property_area == "Urban" else 0
semiurban = 1 if property_area == "Semiurban" else 0
rural = 1 if property_area == "Rural" else 0



if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Urban": urban,
    "Semiurban": semiurban,
    "Rural": rural
}])

    
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
        