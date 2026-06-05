
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/vehicle_model.pkl")

model = load_model()

# App Title
st.title("🚗 Vehicle Traffic Analysis")
st.write("Predict vehicle traffic using the trained machine learning model.")

st.sidebar.header("Input Features")

# User Inputs
feature1 = st.sidebar.number_input("Feature 1", value=10.0)
feature2 = st.sidebar.number_input("Feature 2", value=20.0)
feature3 = st.sidebar.number_input("Feature 3", value=30.0)
feature4 = st.sidebar.number_input("Feature 4", value=40.0)

# Create DataFrame
input_data = pd.DataFrame({
    "Feature1": [feature1],
    "Feature2": [feature2],
    "Feature3": [feature3],
    "Feature4": [feature4]
})

st.subheader("Input Data")
st.write(input_data)

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted Value: {prediction[0]:.2f}")

st.markdown("---")
st.write("Vehicle Traffic Analysis Project")
