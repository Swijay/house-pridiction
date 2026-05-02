import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="centered",
)


st.title("🏠 House Price Prediction Dashboard")

st.write(
    "Enter house details below and get an estimated house price using the trained machine learning model."
)


area = st.number_input("Area", min_value=500, max_value=20000, value=7500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=3)
stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=2)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"],
)


input_data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus,
}


if st.button("Predict House Price"):
    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            predicted_price = response.json()["predicted_price"]

            st.success(f"Estimated House Price: ₹{predicted_price:,.2f}")
        else:
            st.error("API error. Please check if FastAPI server is running.")

    except Exception as e:
        st.error("Could not connect to FastAPI server.")
        st.write(e)