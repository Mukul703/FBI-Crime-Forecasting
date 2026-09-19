import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from xgboost import XGBRegressor

from src.pipeline import create_forecasting_features

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load XGBoost model
model = XGBRegressor()
model.load_model(str(MODEL_DIR / "final_xgb_model.json"))

# Load historical data
historical_data = joblib.load(
    MODEL_DIR / "historical_monthly_crime_data.pkl"
)

# Load metadata
metadata = joblib.load(
    MODEL_DIR / "xgb_model_metadata.pkl"
)

print("Model loaded successfully!")
print("Historical data shape:", historical_data.shape)
print("Required history:", metadata["required_history_months"])



# Streamlit page configuration
st.set_page_config(
    page_title="FBI Crime Forecasting",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.write("**Project:** FBI Crime Investigation")
st.sidebar.write("**Task:** Monthly Crime Forecasting")
st.sidebar.write("**Model:** XGBoost Regressor")
st.sidebar.write("**Forecast Type:** One-step-ahead")
st.sidebar.write("**Historical Data:** 1999–2011")

# Application title
st.title("FBI Crime Investigation - Monthly Crime Forecasting")

# About the Project
st.header("About the Project")

st.write(
    """
    This project analyzes historical crime records and forecasts
    monthly crime incidents using machine learning.

    The XGBoost Regressor uses historical lag features and
    rolling averages to capture temporal crime patterns.

    The application provides historical crime trends, model
    performance metrics, and one-step-ahead monthly forecasts.
    """
)
st.success("Model loaded successfully!")

st.write("This application forecasts monthly crime incidents using XGBoost.")

# Latest historical data
latest_date = historical_data.index[-1]
latest_count = historical_data.iloc[-1]

# Forecasting section
st.header("Monthly Crime Forecast")

if st.button("Predict Next Month's Crime Count"):

    # Create forecasting features
    input_features = create_forecasting_features(
        historical_data
    )

    # Generate prediction
    prediction = model.predict(input_features)[0]

    # Ensure prediction is non-negative and rounded
    prediction = max(0, round(prediction))

    # Determine forecast month
    forecast_month = latest_date + pd.DateOffset(months=1)

    # Display prediction
    st.metric(
        label=f"Predicted Crime Incidents - {forecast_month.strftime('%B %Y')}",
        value=prediction
    )
    # Create downloadable forecast report
    forecast_df = pd.DataFrame({
        "Forecast Month": [forecast_month.strftime('%B %Y')],
        "Predicted Crime Count": [prediction]
    })
    st.download_button(
        label="Download Forecast Report",
        data=forecast_df.to_csv(index=False),
        file_name="crime_forecast.csv",
        mime="text/csv"
    )

    # Historical data overview
st.header("Historical Data Overview")

st.write(f"Latest Available Month: {latest_date.strftime('%B %Y')}")
st.write(f"Actual Crime Count: {int(latest_count)}")

# Historical crime trend
st.header("Historical Monthly Crime Trend")

st.line_chart(historical_data)

# Forecasting disclaimer
st.info(
    "Note: This forecast is based on historical crime patterns "
    "and is intended for analytical and educational purposes. "
    "It should not be used as the sole basis for operational "
    "policing or resource-allocation decisions."
)

# Model Performance
st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "165.69")

with col2:
    st.metric("RMSE", "218.64")

with col3:
    st.metric("R² Score", "0.4916")

st.caption(
    "Metrics are based on one-step-ahead walk-forward validation."
)
# Model Limitations
st.header("Model Limitations")

st.write(
    """
    - The model forecasts monthly crime incidents using historical patterns.
    - Evaluation is based on one-step-ahead walk-forward validation.
    - The model does not establish causation or predict individual crimes.
    - Additional validation is required before operational use.
    """
)

# Forecast Methodology
st.header("Forecast Methodology")

with st.expander("How does the model work?"):
    st.write(
        """
        The application uses an XGBoost Regressor to forecast
        monthly crime incidents.

        The model uses historical time-series features:

        - Lag 1, 2, 3, 6, and 12 months
        - 3-month rolling average
        - 6-month rolling average
        - 12-month rolling average

        The forecast is generated using the latest available
        historical crime data.
        """
    )