"""
01_data_cleaning.py
--------------------
Loads the raw Global Weather Repository CSV, handles missing values and
outliers, engineers time features from `last_updated`, and writes a
cleaned parquet/csv file to data/processed/.

Run:
    python src/01_data_cleaning.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

import joblib

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
import kagglehub

# Download latest version

RAW_PATH = "data/raw/GlobalWeatherRepository.csv"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "weather_clean.csv")
REPORTS_DIR = "reports"



def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(
        f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns."
    )
    return df

def validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure that all required columns exist before processing.
    Raises an error if any required columns are missing.
    """

    required_columns = [
        "last_updated",
        "temperature_celsius",
        "humidity",
        "country",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"The dataset is missing the following required columns: {missing}"
        )

    logger.info("✓ Required columns validated.")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    """
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows. Removing them...")
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dataset now contains {df.shape[0]:,} rows.")
    else:
        logger.info("No duplicate rows found.")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # last_updated is the core time-series key required by the assessment
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df = df.dropna(subset=["last_updated"])
    df["date"] = df["last_updated"].dt.date
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    df["day_of_week"] = df["last_updated"].dt.dayofweek
    df["day"] = df["last_updated"].dt.day

    df["week"] = df["last_updated"].dt.isocalendar().week.astype(int)

    df["quarter"] = df["last_updated"].dt.quarter

    df["is_weekend"] = (
            df["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------
    # Cyclical Time Features
    # -----------------------------
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["dayofweek_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dayofweek_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_report = df.isna().sum()
    missing_report = missing_report[missing_report > 0].sort_values(ascending=False)
    logger.info(
        f"Missing values before cleaning:\n{missing_report}"
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns

    # Numeric: impute with the median per country (falls back to global median)
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df.groupby("country")[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # Categorical: impute with mode
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])

    return df


def handle_outliers(df: pd.DataFrame, cols=None, method="iqr", z_thresh=4.0) -> pd.DataFrame:
    """
    Cap outliers (winsorize) rather than drop rows, since dropping full
    weather records loses correlated fields we need for the other columns.
    """
    total_outliers = 0
    if cols is None:
        cols = [
            "temperature_celsius", "precip_mm", "humidity", "pressure_mb",
            "wind_kph", "gust_kph", "air_quality_PM2.5", "air_quality_PM10",
        ]
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        if method == "iqr":
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:  # z-score
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - z_thresh * std, mean + z_thresh * std
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        total_outliers += n_outliers
        df[col] = df[col].clip(lower, upper)
        logger.info(f"{col}: capped {n_outliers} outliers to [{lower:.2f}, {upper:.2f}]")

    return df, total_outliers

def encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the country column into numerical values and saves the encoder
    for reuse during prediction.
    """
    if "country" not in df.columns:
        return df

    os.makedirs(MODEL_DIR, exist_ok=True)

    encoder = LabelEncoder()

    df["country_encoded"] = encoder.fit_transform(df["country"])

    COUNTRY_ENCODER_PATH = os.path.join(
        MODEL_DIR,
        "country_encoder.pkl"
    )

    joblib.dump(
        encoder,
        COUNTRY_ENCODER_PATH
    )

    logger.info(f"Saved country encoder -> {COUNTRY_ENCODER_PATH}")

    return df

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = load_data(RAW_PATH)
    initial_rows = len(df)

    duplicate_count = df.duplicated().sum()

    df = remove_duplicates(df)

    rows_after_duplicates = len(df)
    validate_columns(df)
    df = parse_dates(df)
    missing_before = df.isna().sum().sum()
    df = handle_missing_values(df)
    missing_after = df.isna().sum().sum()
    df, outlier_count = handle_outliers(df)
    df = encode_country(df)
    df = df.sort_values("last_updated").reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    report_path = os.path.join(REPORTS_DIR, "cleaning_report.txt")

    with open(report_path, "w") as report:
        report.write("WEATHER DATA CLEANING REPORT\n")
        report.write("=" * 40 + "\n\n")

        report.write(f"Initial Rows: {initial_rows}\n")
        report.write(f"Rows After Duplicate Removal: {rows_after_duplicates}\n")
        report.write(f"Duplicates Removed: {duplicate_count}\n\n")

        report.write(f"Missing Values Before Cleaning: {missing_before}\n")
        report.write(f"Missing Values After Cleaning: {missing_after}\n\n")

        report.write(f"Outliers Capped: {outlier_count}\n\n")

        report.write(f"Final Rows: {len(df)}\n")
        report.write(f"Final Columns: {len(df.columns)}\n")
        logger.info(f"Cleaning report saved -> {report_path}")
    PARQUET_PATH = os.path.join(
        OUT_DIR,
        "weather_clean.parquet"
    )

    df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"\nSaved cleaned dataset -> {OUT_PATH}  ({df.shape[0]:,} rows)")
    logger.info(
        f"Saved CSV -> {OUT_PATH}"
    )

    logger.info(
        f"Saved Parquet -> {PARQUET_PATH}"
    )


if __name__ == "__main__":
    main()
