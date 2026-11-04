import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

@st.cache_resource
def train_selected_model():
    """Train the best-performing model on the complete cleaned dataset."""
    data = pd.read_csv("car data.csv")
    data = data.drop_duplicates()
    data["Car_Age"] = 2018 - data["Year"]
    data = data.drop(["Car_Name", "Year"], axis=1)
    data = pd.get_dummies(
        data,
        columns=["Fuel_Type", "Selling_type", "Transmission"],
        drop_first=True,
        dtype=int
    )
    features = data.drop("Selling_Price", axis=1)
    target = data["Selling_Price"]
    selected_model = LinearRegression()
    selected_model.fit(features, target)
    return selected_model

@st.cache_data(ttl=3600)
def get_inr_to_pkr_rate():
    """Fetch a live INR/PKR rate; use a clearly disclosed fallback if offline."""
    fallback_rate = 2.92
    try:
        response = requests.get(
            "https://open.er-api.com/v6/latest/INR",
            timeout=5
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("result") == "success":
            rate = payload.get("rates", {}).get("PKR")
            if rate and rate > 0:
                return float(rate), "live"
    except (requests.RequestException, ValueError, TypeError):
        pass
    return fallback_rate, "fallback"

model = train_selected_model()
inr_to_pkr, rate_source = get_inr_to_pkr_rate()

st.title("🚗 Car Price Prediction")
st.write("Enter the car details to estimate its selling price.")
if rate_source == "live":
    st.caption(f"INR → PKR exchange rate: 1 INR = PKR {inr_to_pkr:.4f} (live reference; cached for 1 hour).")
else:
    st.warning(
        f"Live exchange rate unavailable. Using fallback rate "
        f"1 INR = PKR {inr_to_pkr:.2f}; this may be outdated."
    )

st.divider()
st.header("Enter Car Details")

with st.form("car_prediction_form"):
    present_price = st.number_input(
        "Present Price (dataset unit: INR lakh)",
        min_value=0.32,
        max_value=92.6,
        value=5.0,
        help="Allowed range is based on the training dataset (INR lakh)."
    )
    driven_kms = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=10000, help="The training dataset contains up to 500,000 km.")
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=15, value=5, help="The training dataset covers car ages from 0 to 15 years.")
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Previous Owners", [0, 1, 2, 3])
    submit = st.form_submit_button("Predict Car Price")

if submit:
    if driven_kms < 500:
        st.warning("The training dataset starts at 500 km. This prediction uses an input below the observed training range.")

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
    prediction_inr_lakh = float(model.predict(input_data)[0])
    prediction_inr = prediction_inr_lakh * 100000
    prediction_pkr = prediction_inr * inr_to_pkr
    st.success("Prediction generated")
    st.subheader("Estimated Selling Price")
    st.title(f"PKR {prediction_pkr:,.0f}")
    st.caption(
        f"Approx. INR {prediction_inr:,.0f} before conversion. "
        "Currency conversion is indicative only; actual exchange rates and "
        "Pakistan used-car market prices may differ."
    )
