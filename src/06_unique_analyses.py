"""
06_unique_analyses.py
-----------------------
Advanced "Unique Analyses" requirements:
  - Climate analysis: monthly/seasonal patterns by continent
  - Environmental impact: air quality correlation with weather params
  - Feature importance: Random Forest + permutation importance
  - Spatial analysis: choropleth-style scatter of temperature by lat/lon
  - Geographical patterns: country/continent comparisons

Run:
    python src/06_unique_analyses.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — just saves PNGs, no GUI/Tk needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import logging
import joblib

IN_PATH = "data/processed/weather_clean.csv"
FIG_DIR = "visuals"
REPORT_DIR = "reports"
MODEL_DIR = "models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def climate_analysis(df: pd.DataFrame):
    if "continent" not in df.columns:
        logger.info("No 'continent' column in dataset — skipping continent-level climate analysis.")
        return
    monthly = df.groupby(["continent", "month"])["temperature_celsius"].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="month", y="temperature_celsius", hue="continent", marker="o")
    plt.title("Monthly Average Temperature by Continent")
    plt.xlabel("Month")
    plt.ylabel("Avg Temperature (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "climate_by_continent.png"), dpi=150)
    plt.close()


def air_quality_correlation(df: pd.DataFrame):
    aq_cols = [c for c in df.columns if c.startswith("air_quality")]
    weather_cols = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm"]
    aq_cols = [c for c in aq_cols if c in df.columns]
    weather_cols = [c for c in weather_cols if c in df.columns]
    if not aq_cols:
        logger.info("No air_quality_* columns found — skipping.")
        return
    corr = df[aq_cols + weather_cols].corr().loc[aq_cols, weather_cols]
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Air Quality vs Weather Parameters — Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "air_quality_correlation.png"), dpi=150)
    plt.close()


def feature_importance(df: pd.DataFrame, target="temperature_celsius"):
    candidate_features = [
        "humidity", "pressure_mb", "wind_kph", "cloud", "uv_index",
        "visibility_km", "precip_mm", "latitude", "longitude",
    ]
    features = [c for c in candidate_features if c in df.columns]
    X = df[features].fillna(df[features].median())
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "random_forest_temperature.pkl"
    )

    joblib.dump(
        rf,
        MODEL_PATH
    )

    logger.info(
        f"Saved Random Forest model -> {MODEL_PATH}"
    )

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    importance_df = pd.DataFrame({

        "Feature": features,

        "RF Importance": rf.feature_importances_,

        "Permutation Importance": perm.importances_mean

    })

    importance_df = importance_df.sort_values(

        by="RF Importance",

        ascending=False

    )

    importance_df.to_csv(

        os.path.join(

            REPORT_DIR,

            "feature_importance.csv"

        ),

        index=False

    )

    logger.info(
        "Saved feature importance report."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=axes[0], palette="mako")
    axes[0].set_title("Random Forest — Built-in Feature Importance")
    sns.barplot(x=perm_importances.values, y=perm_importances.index, ax=axes[1], palette="mako")
    axes[1].set_title("Permutation Importance (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Top predictors of temperature (RF importance):\n", importances.head(5))


def climate_risk_score(df):

    columns = [

        "temperature_celsius",

        "humidity",

        "wind_kph",

        "precip_mm"

    ]

    available = [

        c for c in columns

        if c in df.columns

    ]

    risk = df[available].copy()

    risk = (

        risk - risk.min()

    ) / (

        risk.max() - risk.min()

    )

    df["climate_risk_score"] = risk.mean(axis=1)

    ranking = (

        df.groupby("country")["climate_risk_score"]

        .mean()

        .sort_values(ascending=False)

    )

    ranking.to_csv(

        os.path.join(

            REPORT_DIR,

            "climate_risk_rankings.csv"

        )

    )

    logger.info(
        "Saved climate risk rankings."
    )

def extreme_weather(df):

    summary = (

        df.groupby("country")

        .agg({

            "temperature_celsius":"max",

            "precip_mm":"max",

            "wind_kph":"max"

        })

    )

    summary.to_csv(

        os.path.join(

            REPORT_DIR,

            "extreme_weather_leaderboard.csv"

        )

    )

    logger.info(
        "Saved extreme weather leaderboard."
    )


def spatial_analysis(df: pd.DataFrame):
    if not {"latitude", "longitude"}.issubset(df.columns):
        logger.info("No latitude/longitude columns — skipping spatial analysis.")
        return
    snapshot = df.groupby(["country", "latitude", "longitude"])["temperature_celsius"].mean().reset_index()
    plt.figure(figsize=(14, 7))
    sc = plt.scatter(
        snapshot["longitude"], snapshot["latitude"],
        c=snapshot["temperature_celsius"], cmap="coolwarm", s=40, edgecolor="k", linewidth=0.3
    )
    plt.colorbar(sc, label="Avg Temperature (°C)")
    plt.title("Global Spatial Distribution of Average Temperature")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "spatial_temperature_map.png"), dpi=150)
    plt.close()
    snapshot.to_csv(

        os.path.join(

            REPORT_DIR,

            "global_temperature_coordinates.csv"

        ),

        index=False

    )


def geographical_patterns(df: pd.DataFrame):
    if "continent" not in df.columns:
        logger.info("No 'continent' column — skipping geographical pattern comparison.")
        return
    summary = (

        df.groupby("continent")

        .agg({

            "temperature_celsius": ["mean", "max", "min"],

            "humidity": "mean",

            "precip_mm": "mean",

            "wind_kph": "mean"

        })

    )
    summary.to_csv("reports/geographical_summary.csv")
    logger.info("Saved continent-level summary -> reports/geographical_summary.csv")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["last_updated"])
    if "month" not in df.columns:
        df["month"] = df["last_updated"].dt.month

    climate_analysis(df)
    air_quality_correlation(df)
    feature_importance(df)
    climate_risk_score(df)
    extreme_weather(df)
    spatial_analysis(df)
    geographical_patterns(df)
    logger.info(f"\nSaved unique-analysis visuals -> {FIG_DIR}/")
    with open(

            os.path.join(

                REPORT_DIR,

                "unique_analysis_summary.txt"

            ),

            "w"

    ) as report:
        report.write("UNIQUE ANALYSIS SUMMARY\n")

        report.write("=" * 50 + "\n\n")

        report.write(
            "Completed Analyses:\n"
        )

        report.write(
            "- Climate Analysis\n"
        )

        report.write(
            "- Air Quality Correlation\n"
        )

        report.write(
            "- Feature Importance\n"
        )

        report.write(
            "- Climate Risk Ranking\n"
        )

        report.write(
            "- Spatial Analysis\n"
        )

        report.write(
            "- Extreme Weather Leaderboard\n"
        )

        report.write(
            "- Geographical Summary\n"
        )

    logger.info(
        "Saved unique analysis summary."
    )


if __name__ == "__main__":
    main()
