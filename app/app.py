import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from xgboost import XGBRegressor

from src.pipeline import create_forecasting_features


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="FBI Crime Forecasting",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load Model and Data
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(
        str(MODEL_DIR / "final_xgb_model.json")
    )
    return model


@st.cache_data
def load_data():
    historical_data = joblib.load(
        MODEL_DIR / "historical_monthly_crime_data.pkl"
    )

    metadata = joblib.load(
        MODEL_DIR / "xgb_model_metadata.pkl"
    )

    return historical_data, metadata


model = load_model()
historical_data, metadata = load_data()


# --------------------------------------------------
# Important Historical Information
# --------------------------------------------------

latest_date = historical_data.index[-1]
latest_count = historical_data.iloc[-1]

forecast_month = latest_date + pd.DateOffset(months=1)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📊 Project Information")

st.sidebar.markdown(
    """
    **Project:** FBI Crime Investigation

    **Task:** Monthly Crime Forecasting

    **Model:** XGBoost Regressor

    **Forecast Type:** One-step-ahead

    **Historical Data:** 1999–2011
    """
)

st.sidebar.divider()

st.sidebar.subheader("Technology Stack")

st.sidebar.write("• Python")
st.sidebar.write("• Pandas")
st.sidebar.write("• XGBoost")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Joblib")

st.sidebar.divider()

st.sidebar.caption(
    "Developed for analytical and educational purposes."
)


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("📊 FBI Crime Investigation")
st.subheader("Monthly Crime Forecasting Dashboard")

st.markdown(
    """
    This application analyzes historical crime patterns and
    forecasts monthly crime incidents using an XGBoost
    regression model with time-series features.
    """
)

st.success("✅ Model and historical data loaded successfully!")


# --------------------------------------------------
# Dashboard Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Overview",
        "🔮 Forecast",
        "📈 Historical Trends",
        "📊 Model Performance",
        "📘 Methodology"
    ]
)


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tab1:

    st.header("Project Overview")

    st.write(
        """
        The project uses historical monthly crime records
        to identify temporal patterns and generate a
        one-step-ahead crime forecast.
        """
    )

    st.subheader("Historical Data Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latest Available Month",
            latest_date.strftime("%B %Y")
        )

    with col2:
        st.metric(
            "Latest Crime Count",
            f"{int(latest_count):,}"
        )

    with col3:
        st.metric(
            "Historical Months",
            len(historical_data)
        )

    st.divider()

    st.subheader("Historical Monthly Crime Trend")

    st.line_chart(
        historical_data,
        use_container_width=True
    )

    st.info(
        "Historical trends represent recorded monthly crime "
        "counts in the available dataset."
    )


# ==================================================
# TAB 2: FORECAST
# ==================================================

with tab2:

    st.header("Monthly Crime Forecast")

    st.write(
        f"""
        The model uses historical crime counts and
        time-series features to forecast the next month.

        **Forecast Month:** {forecast_month.strftime("%B %Y")}
        """
    )

    if st.button(
        "🔮 Predict Next Month's Crime Count",
        key="forecast_button"
    ):

        # Create forecasting features
        input_features = create_forecasting_features(
            historical_data
        )

        # Generate prediction
        prediction = model.predict(
            input_features
        )[0]

        # Ensure prediction is non-negative and rounded
        prediction = max(0, round(prediction))

        st.success("Forecast generated successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"Predicted Crime Incidents - "
                      f"{forecast_month.strftime('%B %Y')}",
                value=f"{prediction:,}"
            )

        with col2:
            st.metric(
                label="Latest Actual Crime Count",
                value=f"{int(latest_count):,}"
            )

        st.divider()

        # Forecast report
        forecast_df = pd.DataFrame(
            {
                "Forecast Month": [
                    forecast_month.strftime("%B %Y")
                ],
                "Predicted Crime Count": [
                    prediction
                ]
            }
        )

        st.subheader("Forecast Report")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="⬇️ Download Forecast Report",
            data=forecast_df.to_csv(index=False),
            file_name="crime_forecast.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Click the button above to generate the next "
            "month's crime forecast."
        )

    st.warning(
        """
        This forecast is based on historical patterns
        and should not be used as the sole basis for
        operational policing or resource-allocation decisions.
        """
    )


# ==================================================
# TAB 3: HISTORICAL TRENDS
# ==================================================

with tab3:

    st.header("Historical Crime Trends")

    st.write(
        """
        Explore the monthly crime pattern across the
        available historical period.
        """
    )

    st.subheader("Monthly Crime Count")

    st.line_chart(
        historical_data,
        use_container_width=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Monthly Crime",
            f"{historical_data.mean():,.0f}"
        )

    with col2:
        st.metric(
            "Maximum Monthly Crime",
            f"{historical_data.max():,.0f}"
        )

    with col3:
        st.metric(
            "Minimum Monthly Crime",
            f"{historical_data.min():,.0f}"
        )

    st.subheader("Historical Data")

    historical_df = historical_data.to_frame(
        name="Crime Count"
    )

    st.dataframe(
        historical_df,
        use_container_width=True
    )


# ==================================================
# TAB 4: MODEL PERFORMANCE
# ==================================================

with tab4:

    st.header("Model Performance")

    st.write(
        """
        The XGBoost model was evaluated using
        one-step-ahead walk-forward validation.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            "165.69"
        )

    with col2:
        st.metric(
            "RMSE",
            "218.64"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.4916"
        )

    st.caption(
        "Metrics are based on one-step-ahead walk-forward validation."
    )

    st.divider()

    st.subheader("Model Limitations")

    st.markdown(
        """
        - The model forecasts monthly crime incidents using historical patterns.
        - Evaluation is based on one-step-ahead walk-forward validation.
        - The model does not establish causation.
        - The model does not predict individual crimes.
        - Additional validation is required before operational use.
        """
    )


# ==================================================
# TAB 5: METHODOLOGY
# ==================================================

with tab5:

    st.header("Forecast Methodology")

    st.subheader("Machine Learning Model")

    st.write(
        """
        The application uses an XGBoost Regressor
        for monthly crime forecasting.
        """
    )

    st.subheader("Time-Series Features")

    st.markdown(
        """
        The model uses the following historical features:

        - Lag 1 month
        - Lag 2 months
        - Lag 3 months
        - Lag 6 months
        - Lag 12 months
        - 3-month rolling average
        - 6-month rolling average
        - 12-month rolling average
        """
    )

    st.subheader("Forecast Process")

    st.markdown(
        """
        1. Load historical monthly crime data.
        2. Generate lag and rolling-average features.
        3. Prepare the latest historical observations.
        4. Generate the next month's prediction using XGBoost.
        5. Display and export the forecast result.
        """
    )

    st.subheader("Validation Approach")

    st.write(
        """
        Model evaluation was performed using
        one-step-ahead walk-forward validation.
        """
    )