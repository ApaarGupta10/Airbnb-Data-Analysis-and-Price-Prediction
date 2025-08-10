import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

# --- City mapping from notebook ---
city_map = {
    'Paris': 0, 'Rome': 1, 'Amsterdam': 2, 'Berlin': 3, 'Prague': 4,
    'Barcelona': 5, 'Budapest': 6, 'Vienna': 7, 'Athens': 8, 'Istanbul': 9,
    'Dublin': 10, 'Oslo': 11, 'Stockholm': 12, 'Copenhagen': 13, 'Brussels': 14
}

# City coordinates
city_coords = {
    "Paris": (48.8566, 2.3522), "Rome": (41.9028, 12.4964),
    "Amsterdam": (52.3676, 4.9041), "Berlin": (52.5200, 13.4050),
    "Prague": (50.0755, 14.4378), "Barcelona": (41.3851, 2.1734),
    "Budapest": (47.4979, 19.0402), "Vienna": (48.2082, 16.3738),
    "Athens": (37.9838, 23.7275), "Istanbul": (41.0082, 28.9784),
    "Dublin": (53.3498, -6.2603), "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686), "Copenhagen": (55.6761, 12.5683),
    "Brussels": (50.8503, 4.3517)
}

# Room type mapping
room_type_map = {
    'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2, 'Hotel room': 3
}

@st.cache_resource
def load_model_and_resources():
    """Load model, scaler, and neighbourhood frequency mapping, fix tuple-key format."""
    model = joblib.load("lightgbm_airbnb_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    raw_map = joblib.load("neigh_freq_map.pkl")

    # Fix tuple-key map → nested dict
    city_id_to_name = {v: k for k, v in city_map.items()}
    fixed_map = {}
    for key, freq in raw_map.items():
        if isinstance(key, tuple) and len(key) == 2:
            city_id, neigh = key
            city_name = city_id_to_name.get(city_id, city_id)
            fixed_map.setdefault(city_name, {})[neigh] = freq
        elif isinstance(key, str):
            fixed_map[key] = freq
    return model, scaler, fixed_map

model, scaler, neigh_freq_map = load_model_and_resources()

feature_order = [
    'latitude', 'longitude', 'room_type', 'minimum_nights',
    'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
    'availability_365', 'number_of_reviews_ltm', 'city', 'neighbourhood_freq_enc'
]

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("### ℹ️ Model Information")
    st.write(f"**Cities in mapping:** {len(neigh_freq_map)} / {len(city_map)}")
    st.markdown(f"**Available neighbourhoods:** {len(neigh_freq_map)} / {len(city_map)}")
    st.markdown(f"**Cities supported:** {len(city_map)}")
    st.markdown(f"**Features used:** {len(feature_order)}")

    
    st.markdown("---")
    with st.expander("🏘️ Neighbourhood in each City"):
        for city, neighs in neigh_freq_map.items():
            st.write(f"**{city}**: {len(neighs)} neighbourhoods")
    

    with st.expander("📋 Feature List"):
        for i, feature in enumerate(feature_order, 1):
            st.write(f"{i}. {feature}")

    with st.expander("🏙️ City Encodings"):
        for city_name, encoding in city_map.items():
            st.write(f"{city_name}: {encoding}")
    st.markdown("---")

    auto_predict = st.checkbox("🔄 Auto Predict on Change", value=False)
    st.markdown("---")


# --- Main UI ---
st.title("🏠 Airbnb Price Predictor")
st.markdown("### Predict the nightly Airbnb price in European cities based on your inputs.")

col1, col2 = st.columns([1, 2])

with col1:
    city = st.selectbox("🌍 City", sorted(neigh_freq_map.keys()))
    default_lat, default_lon = city_coords.get(city, (0.0, 0.0))
    latitude = st.number_input("📍 Latitude", value=default_lat, format="%.6f")
    longitude = st.number_input("📍 Longitude", value=default_lon, format="%.6f")

    neighbourhoods = list(neigh_freq_map.get(city, {}).keys())
    if not neighbourhoods:
        st.warning(f"No neighbourhood mapping found for {city}. Predictions may be less accurate.")
        neighbourhoods = ["Unknown"]
    
    for i in range(len(neighbourhoods)):
        neighbourhoods[i] = neighbourhoods[i][1]

    neighbourhood = st.selectbox("🏘️ Neighbourhood", sorted(neighbourhoods))
    room_type = st.selectbox("🏠 Room Type", list(room_type_map.keys()))
    minimum_nights = st.slider("🌙 Minimum Nights", 1, 30, 1)
    number_of_reviews = st.slider("⭐ Total Reviews", 0, 500, 0)
    reviews_per_month = st.number_input("📅 Reviews per Month", 0.0, 50.0, step=0.1, format="%.2f")
    calculated_host_listings_count = st.slider("🏘️ Host Listings Count", 1, 10, 1)
    availability_365 = st.slider("📆 Availability (days/year)", 0, 365, 180)
    number_of_reviews_ltm = st.slider("📈 Reviews Last 12 Months", 0, 500, 0)

with col2:
    m = folium.Map(location=[latitude, longitude], zoom_start=13)
    folium.Marker([latitude, longitude], popup=f"{city} - {neighbourhood}").add_to(m)
    st_folium(m, height=400, width=700)

def make_prediction():
    """Run model prediction."""
    enc_neigh = neigh_freq_map.get(city, {}).get(neighbourhood, 0)
    input_df = pd.DataFrame([{
        'latitude': latitude, 'longitude': longitude, 'room_type': room_type_map[room_type],
        'minimum_nights': minimum_nights, 'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month, 'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365, 'number_of_reviews_ltm': number_of_reviews_ltm,
        'city': city_map[city], 'neighbourhood_freq_enc': enc_neigh
    }], columns=feature_order)

    X_scaled = scaler.transform(input_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=input_df.columns)
    pred_price_log = model.predict(X_scaled_df)[0]
    return pred_price_log

# Prediction section
st.markdown("---")
if auto_predict or st.button("💰 Predict Price"):
    price = make_prediction()
    colA, colB, colC = st.columns(3)
    colA.metric("Predicted Price", f"€{price:.0f}")
    colB.metric("Weekly Price", f"€{price*7:.0f}")
    colC.metric("Monthly Price", f"€{price*30:.0f}")
      # Additional insights
    st.markdown("### 📊 Listing Summary")

    summary_data = {
            "Feature": ["City", "Neighbourhood", "Room Type", "Location", "Minimum Stay", "Host Listings", "Availability"],
            "Value": [
                city,
                neighbourhood,
                room_type,
                f"{latitude:.4f}, {longitude:.4f}",
                f"{minimum_nights} night(s)",
                f"{calculated_host_listings_count} listing(s)",
                f"{availability_365} days/year"
            ]
        }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    