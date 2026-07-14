"""
05_multi_model_ensemble.py
----------------------------
Advanced assessment: build and compare multiple forecasting models
(SARIMA, Prophet, XGBoost regression on lag features), then combine
them into a simple average ensemble and compare metrics.

Run:
    python src/05_multi_model_ensemble.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — just saves PNGs, no GUI/Tk needed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import joblib
import logging
from sklearn.metrics import r2_score

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
TEST_SIZE = 14


def build_daily_series(df):
    daily = df.groupby(df["last_updated"].dt.date)["temperature_celsius"].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily.asfreq("D").interpolate()


def sarima_forecast(train, steps):
    model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    joblib.dump(
        fitted,
        os.path.join(
            MODEL_DIR,
            "sarima_ensemble.pkl"
        )
    )

    logger.info("Saved SARIMA model.")
    return fitted.forecast(steps=steps).values


def prophet_forecast(train, steps):
    try:
        from prophet import Prophet
    except ImportError:
        logger.info("Prophet not installed — skipping. `pip install prophet` to enable.")
        return None
    pdf = train.reset_index()
    pdf.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(pdf)
    joblib.dump(
        m,
        os.path.join(
            MODEL_DIR,
            "prophet_model.pkl"
        )
    )

    logger.info("Saved Prophet model.")
    future = m.make_future_dataframe(periods=steps)
    fc = m.predict(future)
    return fc["yhat"].values[-steps:]


def make_lag_features(series: pd.Series, n_lags=7):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()


def xgboost_forecast(train: pd.Series, steps: int, n_lags=7):
    lagged = make_lag_features(train, n_lags)
    X, y = lagged.drop(columns="y"), lagged["y"]
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    importance.to_csv(

        os.path.join(

            REPORT_DIR,

            "xgboost_feature_importance.csv"

        ),

        index=False

    )
    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            "xgboost_model.pkl"
        )
    )

    logger.info("Saved XGBoost model.")

    # iterative multi-step forecast
    history = list(train.values)
    preds = []
    for step in range(steps):
        row = {}
        for lag in range(1, n_lags + 1):
            row[f"lag_{lag}"] = history[-lag]
        next_date = train.index[-1] + pd.Timedelta(days=step + 1)
        row["dow"] = next_date.dayofweek
        row["month"] = next_date.month
        X_next = pd.DataFrame([row])[X.columns]
        pred = model.predict(X_next)[0]
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def evaluate(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        ) * 100
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    logger.info(
        f"{name} | "
        f"MAE={mae:.3f} "
        f"RMSE={rmse:.3f} "
        f"MAPE={mape:.2f}% "
        f"R²={r2:.3f}"
    )

    return {

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "R2": r2

    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["last_updated"])
    series = build_daily_series(df)

    train, test = series.iloc[:-TEST_SIZE], series.iloc[-TEST_SIZE:]

    results = {}
    results["SARIMA"] = sarima_forecast(train, TEST_SIZE)
    prophet_res = prophet_forecast(train, TEST_SIZE)
    if prophet_res is not None:
        results["Prophet"] = prophet_res
    results["XGBoost"] = xgboost_forecast(train, TEST_SIZE)

    # Ensemble = simple average of available models
    stacked = np.vstack(list(results.values()))
    results["Ensemble"] = stacked.mean(axis=0)
    prediction_df = pd.DataFrame({

        "Date": test.index,

        "Actual": test.values

    })

    for name, preds in results.items():
        prediction_df[name] = preds

    prediction_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "all_model_predictions.csv"

        ),

        index=False

    )

    logger.info(
        "Saved model predictions."
    )

    scoreboard = [evaluate(name, test.values, preds) for name, preds in results.items()]
    scoreboard_df = pd.DataFrame(scoreboard).sort_values("RMSE")
    scoreboard_df.to_csv("reports/model_comparison.csv", index=False)
    best = scoreboard_df.iloc[0]

    with open(

            os.path.join(

                REPORT_DIR,

                "best_model.txt"

            ),

            "w"

    ) as f:

        f.write("BEST FORECASTING MODEL\n")

        f.write("=" * 40 + "\n\n")

        f.write(
            f"Model : {best['Model']}\n"
        )

        f.write(
            f"MAE   : {best['MAE']:.3f}\n"
        )

        f.write(
            f"RMSE  : {best['RMSE']:.3f}\n"
        )

        f.write(
            f"MAPE  : {best['MAPE']:.2f}%\n"
        )

        f.write(
            f"R²    : {best['R2']:.3f}\n"
        )

    logger.info(
        "Saved best model report."
    )
    logger.info("\nBest model by RMSE:\n", scoreboard_df.iloc[0])

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
    colors = {

        "SARIMA": "red",

        "Prophet": "green",

        "XGBoost": "orange",

        "Ensemble": "blue"

    }

    for name, preds in results.items():
        plt.plot(

            test.index,

            preds,

            "--",

            linewidth=2,

            color=colors.get(name),

            label=name

        )
    plt.title("Multi-Model Forecast Comparison — Global Avg Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "multi_model_comparison.png"), dpi=150)
    plt.close()
    logger.info(f"\nSaved -> {FIG_DIR}/multi_model_comparison.png, reports/model_comparison.csv")


if __name__ == "__main__":
    main()
