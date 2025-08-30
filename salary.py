import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("salary_model.pkl")

st.title("💼 Salary Prediction App")


# Input fields
age = st.number_input("Age",min_value=18, step=1)
experience = st.number_input("Years of Experience",step=0.5)

input_data = np.array([[age,experience]])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Salary: {prediction[0]:,.2f}")

