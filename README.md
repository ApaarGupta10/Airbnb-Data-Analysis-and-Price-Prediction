# 🏠 Airbnb Price Prediction – Streamlit + LightGBM

## 📌 Overview

This project predicts the nightly rental price of Airbnb listings across major European cities using Machine Learning (LightGBM).
It combines exploratory data analysis (EDA), feature engineering, and an interactive Streamlit web app with maps and visualizations.

## 🎯 Problem Statement

Airbnb pricing is inconsistent due to location, property type, demand, and seasonal factors.

Guests may overpay or underpay due to lack of transparent pricing.

Hosts face difficulties in setting competitive but profitable prices.

## ✅ Solution

Built a LightGBM regression model trained on preprocessed Airbnb datasets.

Applied neighbourhood frequency encoding and feature scaling.

Developed an interactive Streamlit app with:

City & neighbourhood selection

Room type, availability, and host listing details

Dynamic map visualization (Folium)

Predicted price with weekly & monthly estimates

## 🛠️ Tech Stack

Python: pandas, numpy, scikit-learn, LightGBM

Visualization: matplotlib, seaborn, folium

Deployment: Streamlit, joblib

Database (optional): Google BigQuery (for preprocessed data storage)

## 🚀 Features

Predicts nightly, weekly, and monthly price.

Interactive map preview for selected location.

Supports 15 European cities (Paris, Rome, Amsterdam, etc.).

Shows encoded values (city, room type, neighbourhood frequency).

Error handling with detailed debug info.

## 📂 Project Structure

#### ├── app.py                         # Streamlit app

#### ├── AirBNB_Data_Analysis_...ipynb  # Jupyter notebook with EDA + model training

#### ├── lightgbm_airbnb_model.pkl      # Trained ML model

#### ├── feature_scaler.pkl             # Scaler for preprocessing

#### ├── neigh_freq_map.pkl             # Neighbourhood frequency encoding

#### ├── Listings.csv                   # Raw dataset

#### ├── requirements.txt               # Python dependencies

#### └── README.md                      # Project documentation

## ⚡ Installation & Usage

### Clone the repo:

git clone https://github.com/your-username/airbnb-price-prediction.git
cd airbnb-price-prediction


### Install dependencies:

pip install -r requirements.txt


### Run the app:

streamlit run app.py

## 📊 Results & Insights

Achieved strong predictive performance using LightGBM.

Neighbourhood frequency encoding improved feature representation.

Streamlit app makes predictions transparent and user-friendly.

## 🌍 Future Scope

Integrate real-time Airbnb API data.

Deploy app on Streamlit Cloud / GCP / AWS.

Enhance prediction with seasonality & holiday demand.

Build recommendation system for hosts (e.g., "optimal price suggestions").

## 🙌 Contributors
Apaar Gupta and Shubham Gavhane
Link: https://airbnb-data-analysis-and-price-prediction-6abyz3mtthshxt5uvkvv.streamlit.app/
