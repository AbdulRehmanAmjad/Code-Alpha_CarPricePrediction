import streamlit as st
import pandas as pd
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

model = train_selected_model()

# Indicative mid-market reference rate on 4 Nov 2026.
# This is a display conversion, not a Pakistan-market price model.
INR_TO_PKR = 2.92

st.title("🚗 Car Price Prediction")
st.write("Enter the car details to estimate its selling price.")

st.info(
    "Dataset note: the source dataset is commonly interpreted in Indian "
    "Rupees lakhs. The PKR figure is an approximate currency conversion, "
    "not a Pakistan-market valuation."
)
st.divider()
st.header("Enter Car Details")

with st.form("car_prediction_form"):
    present_price = st.number_input(
        "Present Price (dataset unit: INR lakh)",
        min_value=0.1,
        value=5.0,
        help="Use the same unit as the source dataset."
    )
    driven_kms = st.number_input("Kilometers Driven", min_value=0, value=10000)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=50, value=5)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Previous Owners", [0, 1, 2, 3])
    submit = st.form_submit_button("Predict Car Price")

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
    prediction_inr_lakh = float(model.predict(input_data)[0])
    prediction_inr = prediction_inr_lakh * 100000
    prediction_pkr = prediction_inr * INR_TO_PKR
    st.success("Prediction generated")
    st.subheader("Estimated Selling Price")
    st.title(f"PKR {prediction_pkr:,.0f}")
    st.caption(
        f"Approx. INR {prediction_inr:,.0f} before conversion. "
        f"Conversion reference: 1 INR ≈ PKR {INR_TO_PKR:.2f} "
        "(4 Nov 2026; update this rate when needed)."
    )
