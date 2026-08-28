import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("airbnb_final_pipeline_compressed.pkl")

try:
    model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Title
st.title("🏠 Airbnb Price Prediction")
st.write("Predict the Airbnb listing price using the selected features.")

st.divider()

# Input Section
st.subheader("Enter Listing Details")

col1, col2 = st.columns(2)

with col1:
    host_response_time = st.selectbox(
        "Host Response Time",
        ["within an hour", "within a few hours", "within a day", "a few days or more"]
    )

    host_response_rate = st.number_input(
        "Host Response Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

    host_acceptance_rate = st.number_input(
        "Host Acceptance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )

    property_type = st.text_input(
        "Property Type",
        value="Apartment"
    )

    accommodates = st.number_input(
        "Accommodates",
        min_value=1,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        value=1.0
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        value=1
    )

with col2:
    beds = st.number_input(
        "Beds",
        min_value=0,
        value=1
    )

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        value=1
    )

    availability_365 = st.number_input(
        "Availability (365 days)",
        min_value=0,
        max_value=365,
        value=100
    )

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=10
    )

    review_scores_rating = st.number_input(
        "Review Scores Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.5
    )

    instant_bookable = st.selectbox(
        "Instant Bookable",
        ["t", "f"]
    )

    reviews_per_month = st.number_input(
        "Reviews Per Month",
        min_value=0.0,
        value=1.0
    )

st.divider()

# Prediction
if st.button("Predict Price", type="primary"):

    input_data = pd.DataFrame({
        "host_response_time": [host_response_time],
        "host_response_rate": [host_response_rate],
        "host_acceptance_rate": [host_acceptance_rate],
        "room_type": [room_type],
        "property_type": [property_type],
        "accommodates": [accommodates],
        "bathrooms": [bathrooms],
        "bedrooms": [bedrooms],
        "beds": [beds],
        "minimum_nights": [minimum_nights],
        "availability_365": [availability_365],
        "number_of_reviews": [number_of_reviews],
        "review_scores_rating": [review_scores_rating],
        "instant_bookable": [instant_bookable],
        "reviews_per_month": [reviews_per_month]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🏠 Predicted Airbnb Price: {prediction[0]:.2f}"
    )
