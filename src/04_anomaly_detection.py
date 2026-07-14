"""
04_anomaly_detection.py
------------------------
Advanced EDA: anomaly detection on weather features using Isolation
Forest and a rolling z-score method, with visualization.

Run:
    python src/04_anomaly_detection.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — just saves PNGs, no GUI/Tk needed
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib
import logging

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

FEATURES = [
    "temperature_celsius", "precip_mm", "humidity", "pressure_mb",
    "wind_kph", "gust_kph",
]


def isolation_forest_anomalies(df: pd.DataFrame, contamination=0.02):
    feats = [c for c in FEATURES if c in df.columns]
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    df["anomaly_iforest"] = iso.fit_predict(df[feats])  # -1 = anomaly
    df["anomaly_score"] = iso.decision_function(
        df[feats]
    )
    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "isolation_forest.pkl"
    )

    joblib.dump(
        iso,
        MODEL_PATH
    )

    logger.info(
        f"Isolation Forest model saved -> {MODEL_PATH}"
    )
    n_anom = (df["anomaly_iforest"] == -1).sum()
    logger.info(
        f"Isolation Forest flagged {n_anom:,} anomalies ({n_anom / len(df) * 100:.2f}% of rows)"
    )

    return df, n_anom

def compare_contamination_levels(df: pd.DataFrame):

    feats = [c for c in FEATURES if c in df.columns]

    results = []

    for contamination in [0.01, 0.02, 0.05]:

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        )

        labels = model.fit_predict(df[feats])

        results.append({
            "Contamination": contamination,
            "Anomalies": (labels == -1).sum()
        })

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        os.path.join(
            REPORT_DIR,
            "anomaly_model_comparison.csv"
        ),
        index=False
    )

    logger.info(
        "Saved anomaly model comparison."
    )


def rolling_zscore_anomalies(df: pd.DataFrame, col="temperature_celsius", window=30, thresh=3.0):
    daily = df.groupby(df["last_updated"].dt.date)[col].mean()
    daily.index = pd.to_datetime(daily.index)
    roll_mean = daily.rolling(window, min_periods=5).mean()
    roll_std = daily.rolling(window, min_periods=5).std()
    z = (daily - roll_mean) / roll_std
    anomalies = daily[z.abs() > thresh]
    logger.info(f"Rolling z-score flagged {len(anomalies)} anomalous days on '{col}'")
    return daily, anomalies

def plot_anomalies(daily, anomalies):

    plt.figure(figsize=(13, 5))

    plt.plot(
        daily.index,
        daily.values,
        color="steelblue",
        linewidth=2,
        label="Temperature"
    )

    plt.fill_between(
        daily.index,
        daily.values.min(),
        daily.values.max(),
        where=daily.index.isin(anomalies.index),
        alpha=0.25,
        color="red",
        label="Anomaly Period"
    )

    plt.scatter(
        anomalies.index,
        anomalies.values,
        color="red",
        s=60,
        zorder=5
    )

    plt.title("Temperature Anomaly Detection")

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "anomaly_detection.png"
        ),
        dpi=150
    )

    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["last_updated"])

    df, n_anom = isolation_forest_anomalies(df)
    compare_contamination_levels(df)
    anomalies_df = df[
        df["anomaly_iforest"] == -1
        ]

    anomalies_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "anomalies.csv"

        ),

        index=False

    )

    top20 = (

        anomalies_df

        .sort_values(

            by="anomaly_score"

        )

        .head(20)

    )

    columns = [

        "last_updated",

        "city",

        "country",

        "temperature_celsius",

        "humidity",

        "pressure_mb",

        "wind_kph",

        "anomaly_score"

    ]

    columns = [

        c

        for c in columns

        if c in top20.columns

    ]

    top20[columns].to_csv(

        os.path.join(

            REPORT_DIR,

            "top20_anomalies.csv"

        ),

        index=False

    )

    logger.info(
        "Saved Top 20 anomalies."
    )

    logger.info(
        "Saved anomalous records -> reports/anomalies.csv"
    )
    daily, anomalies = rolling_zscore_anomalies(df)
    plot_anomalies(daily, anomalies)
    report_path = os.path.join(
        REPORT_DIR,
        "anomaly_summary.txt"
    )

    with open(report_path, "w") as report:
        report.write("ANOMALY DETECTION REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write(
            f"Total Weather Records: {len(df):,}\n"
        )

        report.write(
            f"Isolation Forest Anomalies: {n_anom:,}\n"
        )

        report.write(
            f"Isolation Forest Percentage: "
            f"{100 * n_anom / len(df):.2f}%\n\n"
        )

        report.write(
            f"Rolling Z-Score Anomalous Days: "
            f"{len(anomalies)}\n"
        )

        report.write(
            f"Rolling Window: 30 days\n"
        )

        report.write(
            f"Z-score Threshold: ±3\n"
        )

    logger.info(
        f"Anomaly report saved -> {report_path}"
    )

    statistics = pd.DataFrame({

        "Metric": [

            "Total Records",

            "Isolation Forest Anomalies",

            "Rolling Z-score Days"

        ],

        "Value": [

            len(df),

            n_anom,

            len(anomalies)

        ]

    })

    statistics.to_csv(

        os.path.join(

            REPORT_DIR,

            "anomaly_statistics.csv"

        ),

        index=False

    )

    logger.info(
        "Saved anomaly statistics."
    )

    df.to_csv("data/processed/weather_with_anomalies.csv", index=False)
    logger.info(f"\nSaved -> visuals/anomaly_detection.png and data/processed/weather_with_anomalies.csv")


if __name__ == "__main__":
    main()
