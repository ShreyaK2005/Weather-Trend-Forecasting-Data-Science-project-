"""
03_basic_forecasting.py
------------------------
Basic assessment requirement: build one forecasting model on the daily
global-average temperature time series (indexed by `last_updated`) and
evaluate it with standard regression/forecast metrics.

Model: SARIMA (statsmodels) — a standard, explainable classical baseline.

Run:
    python src/03_basic_forecasting.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import logging
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import joblib
matplotlib.use("Agg")  # non-interactive backend — just saves PNGs, no GUI/Tk needed
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

IN_PATH = "data/processed/weather_clean.csv"
FIG_DIR = "visuals"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def build_daily_series(
        df: pd.DataFrame,
        city=None
):

    if city is not None:

        if "location_name" not in df.columns:
            raise ValueError("Dataset does not contain a city column.")

        city_df = df[
            df["location_name"].str.lower() == city.lower()
        ]

        if city_df.empty:
            raise ValueError(f"No records found for {city}")

        daily = city_df.groupby(
            city_df["last_updated"].dt.date
        )["temperature_celsius"].mean()

        logger.info(
            f"Forecasting city: {city}"
        )

    else:

        logger.info(
            "Forecasting Global Average Temperature"
        )

        daily = df.groupby(
            df["last_updated"].dt.date
        )["temperature_celsius"].mean()

    daily.index = pd.to_datetime(daily.index)

    daily = daily.asfreq("D").interpolate()

    return daily


def train_test_split_series(series: pd.Series, test_size: int = 14):
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def fit_sarima(train: pd.Series):
    model = SARIMAX(
        train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),  # weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "sarima_model.pkl"
    )

    joblib.dump(
        fitted,
        MODEL_PATH
    )

    logger.info(
        f"SARIMA model saved to {MODEL_PATH}"
    )

    return fitted


def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = (
        mean_absolute_percentage_error(
            y_true,
            y_pred
        ) * 100
    )

    r2 = r2_score(y_true, y_pred)
    metrics = {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,
        "R2": r2

    }

    logger.info(f"MAE : {mae:.3f} °C")
    logger.info(f"RMSE: {rmse:.3f} °C")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"R² : {r2:.3f}")

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "forecast_metrics.csv"

        ),

        index=False

    )

    logger.info(
        "Forecast metrics saved."
    )

    return metrics


def plot_forecast(
        train,
        test,
        forecast,
        conf_int
):

    plt.figure(figsize=(12,5))

    plt.plot(
        train.index[-60:],
        train.values[-60:],
        label="Train"
    )

    plt.plot(
        test.index,
        test.values,
        label="Actual",
        color="black"
    )

    plt.plot(
        test.index,
        forecast,
        label="Forecast",
        color="red",
        linestyle="--"
    )

    plt.fill_between(

        test.index,

        conf_int.iloc[:,0],

        conf_int.iloc[:,1],

        color="red",

        alpha=0.2,

        label="95% Confidence Interval"

    )

    plt.legend()

    plt.title(
        "SARIMA Forecast"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            FIG_DIR,
            "basic_forecast_sarima.png"
        ),

        dpi=150

    )

    plt.close()




def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["last_updated"])
    CITY = None
    if CITY is None:
        logger.info("Generating global weather forecast.")
    else:
        logger.info(f"Generating forecast for city: {CITY}")

    # Examples:
    # CITY = "Nagpur"
    # CITY = "London"

    series = build_daily_series(
        df,
        city=CITY
    )

    train, test = train_test_split_series(series, test_size=14)
    fitted = fit_sarima(train)

    forecast_result = fitted.get_forecast(
        steps=len(test)
    )

    forecast = forecast_result.predicted_mean

    conf_int = forecast_result.conf_int()
    forecast_df = pd.DataFrame({

        "Date": test.index,

        "Actual": test.values,

        "Forecast": forecast.values,

        "Lower95":

            conf_int.iloc[:, 0].values,

        "Upper95":

            conf_int.iloc[:, 1].values

    })

    forecast_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "forecast_results.csv"

        ),

        index=False

    )

    logger.info(
        "Forecast results saved."
    )
    metrics = evaluate(test.values, forecast.values)
    plot_forecast(train, test, forecast,conf_int)

    # ---------- NEW: Forecast Next 30 Days ----------
    future = fitted.get_forecast(steps=30)

    future_dates = pd.date_range(
        start=test.index[-1] + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future.predicted_mean.values,
        "Lower95": future.conf_int().iloc[:, 0].values,
        "Upper95": future.conf_int().iloc[:, 1].values,
    })

    future_df.to_csv(
        os.path.join(REPORT_DIR, "future_forecast.csv"),
        index=False,
    )




    logger.info("30-day future forecast saved.")

    logger.info(f"Forecast plot saved -> {FIG_DIR}")
    logger.info(f"\nSaved forecast plot -> {FIG_DIR}/basic_forecast_sarima.png")



    return metrics





if __name__ == "__main__":
    main()
