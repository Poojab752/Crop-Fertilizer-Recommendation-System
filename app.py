import streamlit as st
import pickle
import numpy as np

# Load models
crop_model = pickle.load(open("models/crop_model.pkl", "rb"))
fertilizer_model = pickle.load(open("models/fertilizer_model.pkl", "rb"))

st.title("🌾 Crop & Fertilizer Recommendation System")

st.header("Enter Soil & Weather Details")

N = st.number_input("Nitrogen", min_value=0)
P = st.number_input("Phosphorous", min_value=0)
K = st.number_input("Potassium", min_value=0)
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")
moisture = st.number_input("Moisture")

if st.button("Predict"):

    crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    crop_pred = crop_model.predict(crop_input)[0]

    fert_input = np.array([[temp, humidity, moisture, N, K, P]])
    fert_pred = fertilizer_model.predict(fert_input)[0]

    st.success(f"Recommended Crop: {crop_pred}")
    st.success(f"Recommended Fertilizer: {fert_pred}")
