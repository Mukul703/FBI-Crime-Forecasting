import pandas as pd


def create_forecasting_features(history):
    """
    Create lag and rolling-average features
    required by the XGBoost forecasting model.
    """

    if len(history) < 12:
        raise ValueError(
            "At least 12 months of historical data are required."
        )

    features = {
        "Lag_1": history.iloc[-1],
        "Lag_2": history.iloc[-2],
        "Lag_3": history.iloc[-3],
        "Lag_6": history.iloc[-6],
        "Lag_12": history.iloc[-12],
        "Rolling_Mean_3": history.iloc[-3:].mean(),
        "Rolling_Mean_6": history.iloc[-6:].mean(),
        "Rolling_Mean_12": history.iloc[-12:].mean()
    }

    return pd.DataFrame([features])