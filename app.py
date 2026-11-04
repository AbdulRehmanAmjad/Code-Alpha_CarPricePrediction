
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Application Title
st.title("🚗 Car Price Prediction")

st.write(
    "Welcome! Enter your car details below "
    "to estimate its selling price."
)

st.divider()

# Car Details Form
st.header("Enter Car Details")

with st.form("car_prediction_form"):

    present_price = st.number_input(
        "Present Price",
        min_value=0.1,
        value=5.0
    )

    driven_kms = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=10000
    )

    car_age = st.number_input(
        "Car Age",
        min_value=0,
        max_value=50,
        value=5
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    selling_type = st.selectbox(
        "Selling Type",
        ["Dealer", "Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3]
    )

    submit = st.form_submit_button(
        "Predict Car Price"
    )

# Prediction Logic
if submit:

    input_data = pd.DataFrame([{
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Owner": owner,
        "Car_Age": car_age,
        "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
        "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
        "Selling_type_Individual": 1 if selling_type == "Individual" else 0,
        "Transmission_Manual": 1 if transmission == "Manual" else 0
    }])

    prediction = model.predict(input_data)
    predicted_price_inr = prediction[0] * 100000
    predicted_price_pkr = predicted_price_inr * 2.91
    st.success("Prediction Successful!")
    st.subheader("Estimated Car Selling Price")
    st.title(f"PKR {predicted_price_pkr:,.0f}")
    st.caption("Approximate price in Pakistani Rupees")     
