import os
import subprocess
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Weather Forecasting Dashboard",
    page_icon="🌦",
    layout="wide"
)

st.title("🌦 AI Weather Forecasting Dashboard")
st.write(
    "This dashboard demonstrates the complete AI weather analytics pipeline "
    "developed for the assessment."
)
st.info(
    """
**About PM Accelerator**

PM Accelerator helps aspiring and experienced Product Managers accelerate
their careers through industry-focused training, AI product development,
leadership programs, interview preparation, and hands-on learning experiences.
"""
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "weather_clean.csv"
)

REPORT_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)

VISUAL_DIR = os.path.join(
    PROJECT_DIR,
    "visuals"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select",

    [

        "Dataset Overview",

        "Forecasting",

        "Anomaly Detection",

        "Unique Analyses",

        "Run Complete Pipeline"

    ]

)

# =========================================================
# DATASET
# =========================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    df = pd.read_csv(DATA_PATH)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Countries", df["country"].nunique())
    c4.metric("Cities", df["location_name"].nunique())

    st.write("### Sample Data")

    st.dataframe(df.head(20), use_container_width=True)

    st.write("### Dataset Columns")

    st.write(list(df.columns))

# =========================================================
# FORECASTING
# =========================================================

elif page == "Forecasting":

    st.header("Temperature Forecasting")

    metrics_path = os.path.join(

        REPORT_DIR,

        "forecast_metrics.csv"

    )

    if os.path.exists(metrics_path):

        metrics = pd.read_csv(metrics_path)

        st.subheader("Forecast Metrics")

        st.dataframe(metrics, use_container_width=True)

    image = os.path.join(

        VISUAL_DIR,

        "basic_forecast_sarima.png"

    )

    if os.path.exists(image):

        st.image(image)

    future = os.path.join(

        REPORT_DIR,

        "future_forecast.csv"

    )

    if os.path.exists(future):

        st.subheader("30-Day Forecast")

        st.dataframe(

            pd.read_csv(future),

            use_container_width=True

        )

# =========================================================
# ANOMALY
# =========================================================

elif page == "Anomaly Detection":

    st.header("Weather Anomaly Detection")

    image = os.path.join(

        VISUAL_DIR,

        "anomaly_detection.png"

    )

    if os.path.exists(image):

        st.image(image)

    anomalies = os.path.join(

        REPORT_DIR,

        "top20_anomalies.csv"

    )

    if os.path.exists(anomalies):

        st.subheader("Top 20 Most Severe Anomalies")

        st.dataframe(

            pd.read_csv(anomalies),

            use_container_width=True

        )

    stats = os.path.join(

        REPORT_DIR,

        "anomaly_statistics.csv"

    )

    if os.path.exists(stats):

        st.subheader("Statistics")

        st.dataframe(

            pd.read_csv(stats),

            use_container_width=True

        )

# =========================================================
# UNIQUE ANALYSIS
# =========================================================

elif page == "Unique Analyses":

    st.header("Unique Analyses")

    figures = [

        "climate_by_continent.png",

        "air_quality_correlation.png",

        "feature_importance.png",

        "spatial_temperature_map.png"

    ]

    for fig in figures:

        path = os.path.join(

            VISUAL_DIR,

            fig

        )

        if os.path.exists(path):

            st.image(path)

    leaderboard = os.path.join(

        REPORT_DIR,

        "extreme_weather_leaderboard.csv"

    )

    if os.path.exists(leaderboard):

        st.subheader("Extreme Weather Leaderboard")

        st.dataframe(

            pd.read_csv(leaderboard),

            use_container_width=True

        )

# =========================================================
# RUN PIPELINE
# =========================================================

else:

    st.header("Run Entire AI Pipeline")

    st.write(
        "This will execute preprocessing, forecasting, anomaly detection, "
        "ensemble modelling and unique analyses."
    )

    if st.button("Run Pipeline"):

        with st.spinner("Running..."):

            subprocess.run(

                [

                    "python",

                    os.path.join(

                        BASE_DIR,

                        "run_all.py"

                    )

                ]

            )

        st.success("Pipeline Completed Successfully!")

        st.balloons()

        st.info(
            "Refresh the dashboard pages to view the newly generated results."
        )
