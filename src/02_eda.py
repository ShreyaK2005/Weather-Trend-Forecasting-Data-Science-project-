"""
02_eda.py
---------
Basic exploratory data analysis: summary stats, correlation heatmap,
temperature/precipitation trend plots, and top-country comparisons.

Run:
    python src/02_eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
import plotly.io as pio
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

IN_PATH = "data/processed/weather_clean.csv"
FIG_DIR = "visuals"
PLOTLY_DIR = "visuals/interactive"
REPORT_DIR = "reports"

sns.set_theme(style="whitegrid")


def load():
    df = pd.read_csv(IN_PATH, parse_dates=["last_updated"])
    return df


def summary_stats(df: pd.DataFrame):
    print(df[["temperature_celsius", "precip_mm", "humidity", "pressure_mb", "wind_kph"]].describe())

def generate_eda_report(df: pd.DataFrame) -> None:
    """
    Generate a text report summarizing key dataset statistics
    and save it to reports/eda_summary.txt.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)

    report_path = os.path.join(REPORT_DIR, "eda_summary.txt")

    total_rows = len(df)
    total_columns = len(df.columns)

    num_countries = (
        df["country"].nunique()
        if "country" in df.columns
        else 0
    )

    num_cities = (
        df["location_name"].nunique()
        if "location_name" in df.columns
        else 0
    )

    avg_temp = df["temperature_celsius"].mean()
    max_temp = df["temperature_celsius"].max()
    min_temp = df["temperature_celsius"].min()

    avg_precip = (
        df["precip_mm"].mean()
        if "precip_mm" in df.columns
        else 0
    )

    avg_humidity = (
        df["humidity"].mean()
        if "humidity" in df.columns
        else 0
    )

    avg_pressure = (
        df["pressure_mb"].mean()
        if "pressure_mb" in df.columns
        else 0
    )

    numeric_df = df.select_dtypes(include="number")

    corr_matrix = numeric_df.corr()

    corr_pairs = (
        corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        .stack()
        .sort_values(key=lambda x: abs(x), ascending=False)
    )

    strongest_pair = corr_pairs.index[0]
    strongest_value = corr_pairs.iloc[0]

    with open(report_path, "w") as report:

        report.write("GLOBAL WEATHER DATASET - EDA REPORT\n")
        report.write("=" * 55 + "\n\n")

        report.write("DATASET OVERVIEW\n")
        report.write("-" * 30 + "\n")

        report.write(f"Rows: {total_rows:,}\n")
        report.write(f"Columns: {total_columns}\n")
        report.write(f"Countries: {num_countries}\n")
        report.write(f"Cities: {num_cities}\n\n")

        report.write("TEMPERATURE STATISTICS\n")
        report.write("-" * 30 + "\n")

        report.write(f"Average Temperature: {avg_temp:.2f} °C\n")
        report.write(f"Maximum Temperature: {max_temp:.2f} °C\n")
        report.write(f"Minimum Temperature: {min_temp:.2f} °C\n\n")

        report.write("OTHER WEATHER STATISTICS\n")
        report.write("-" * 30 + "\n")

        report.write(f"Average Humidity: {avg_humidity:.2f}%\n")
        report.write(f"Average Precipitation: {avg_precip:.2f} mm\n")
        report.write(f"Average Pressure: {avg_pressure:.2f} mb\n\n")

        report.write("STRONGEST CORRELATION\n")
        report.write("-" * 30 + "\n")

        report.write(
            f"{strongest_pair[0]} <-> {strongest_pair[1]}\n"
        )

        report.write(
            f"Correlation: {strongest_value:.3f}\n"
        )

    print(f"EDA report saved -> {report_path}")

def missing_values_heatmap(df: pd.DataFrame) -> None:
    """
    Visualize missing values after preprocessing.
    """
    missing_count = df.isnull().sum().sum()

    logger.info(
        f"Remaining missing values: {missing_count}"
    )

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
        cmap="viridis"
    )

    plt.title("Missing Values Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "missing_values_heatmap.png"),
        dpi=150
    )

    plt.close()


def correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "temperature_celsius", "precip_mm", "humidity", "pressure_mb",
        "wind_kph", "cloud", "uv_index", "visibility_km",
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu",

        aspect="auto",

        title="Correlation Matrix"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "correlation_heatmap.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "correlation_heatmap.png"
        )
    )


def temperature_trend(df: pd.DataFrame) -> None:

    daily = (
        df.groupby(df["last_updated"].dt.date)["temperature_celsius"]
        .mean()
        .reset_index()
    )

    daily.rename(
        columns={
            "last_updated": "Date",
            "temperature_celsius": "Temperature"
        },
        inplace=True
    )

    fig = px.line(
        daily,
        x="Date",
        y="Temperature",
        title="Global Average Temperature Over Time",
        markers=True,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "temperature_trend.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "temperature_trend.png"
        )
    )

def monthly_temperature(df: pd.DataFrame) -> None:
    """
    Average monthly temperature to visualize seasonality.
    """

    monthly = (
        df.groupby("month")["temperature_celsius"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="month",
        y="temperature_celsius",
        markers=True,
        title="Average Temperature by Month"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Average Temperature (°C)"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "monthly_temperature.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "monthly_temperature.png"
        )
    )

def temperature_distribution(df):

    fig = px.histogram(
        df,
        x="temperature_celsius",
        nbins=40,
        marginal="box",
        title="Temperature Distribution"
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "temperature_distribution.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "temperature_distribution.png"
        )
    )

def weather_boxplots(df: pd.DataFrame) -> None:
    """
    Create boxplots for major weather variables to visualize
    the effect of outlier handling.
    """

    columns = [
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph"
    ]

    available = [c for c in columns if c in df.columns]

    fig = px.box(
        df,
        y=available,
        points="outliers",
        title="Distribution of Major Weather Variables"
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "weather_boxplots.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "weather_boxplots.png"
        )
    )

def precipitation_trend(df: pd.DataFrame):

    daily = (
        df.groupby(df["last_updated"].dt.date)["precip_mm"]
        .mean()
        .reset_index()
    )

    daily.rename(
        columns={
            "last_updated": "Date",
            "precip_mm": "Precipitation"
        },
        inplace=True
    )

    fig = px.line(
        daily,
        x="Date",
        y="Precipitation",
        title="Global Average Precipitation",
        markers=True,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "precipitation_trend.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "precipitation_trend.png"
        )
    )


def top_countries_by_temp(df: pd.DataFrame, n=15):

    top = (
        df.groupby("country")["temperature_celsius"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    fig = px.bar(
        top,
        x="temperature_celsius",
        y="country",
        orientation="h",
        color="temperature_celsius",
        title=f"Top {n} Hottest Countries (Average Temperature)"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Average Temperature (°C)",
        yaxis_title="Country"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "top_countries_temperature.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "top_countries_temperature.png"
        )
    )


def temp_vs_humidity(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    fig = px.scatter(

        df.sample(
            min(
                5000,
                len(df)
            )
        ),

        x="humidity",

        y="temperature_celsius",

        color="precip_mm",

        hover_data=[
            "country",
            "location_name"
        ],

        title="Temperature vs Humidity"
    )

    fig.write_html(
        os.path.join(
            PLOTLY_DIR,
            "temp_vs_humidity.html"
        )
    )

    fig.write_image(
        os.path.join(
            FIG_DIR,
            "temp_vs_humidity.png"
        )
    )

def aqi_relationships(df: pd.DataFrame) -> None:
    """
    Explore relationships between AQI and weather parameters.
    """

    relationships = [

        ("temperature_celsius", "Temperature"),

        ("humidity", "Humidity"),

        ("precip_mm", "Precipitation")

    ]

    if "air_quality_PM2.5" not in df.columns:
        return

    for column, label in relationships:

        if column not in df.columns:
            continue

        fig = px.scatter(

            df.sample(
                min(5000, len(df))
            ),

            x=column,

            y="air_quality_PM2.5",

            trendline="ols",

            title=f"PM2.5 vs {label}"

        )

        filename = f"aqi_vs_{column}"

        fig.write_html(

            os.path.join(

                PLOTLY_DIR,

                filename + ".html"

            )

        )

        fig.write_image(

            os.path.join(

                FIG_DIR,

                filename + ".png"

            )

        )


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PLOTLY_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load()
    summary_stats(df)
    missing_values_heatmap(df)
    generate_eda_report(df)
    correlation_heatmap(df)
    temperature_trend(df)
    monthly_temperature(df)
    temperature_distribution(df)
    weather_boxplots(df)
    precipitation_trend(df)
    top_countries_by_temp(df)
    temp_vs_humidity(df)
    aqi_relationships(df)
    print(f"\nSaved EDA visuals -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
