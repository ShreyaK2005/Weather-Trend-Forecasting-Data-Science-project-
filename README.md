# Weather-Trend-Forecasting-Data-Science-project-
Global Weather Trend Forecasting
A data science project analyzing the Global Weather Repository dataset to uncover weather trends and forecast future conditions, completed as part of the PM Accelerator technical assessment.

🌍 About PM Accelerator
PM Accelerator mission statement: [Paste the current, exact mission statement text copied from the PM Accelerator LinkedIn page or pmaccelerator.io here before submitting.]

This is left as a placeholder deliberately — company mission statements get updated, so copy the live text directly from their official page rather than relying on a secondhand paraphrase, to make sure what you submit is accurate and current.

Project Structure
weather-forecasting-assessment/
├── data/
│   ├── raw/                  # Place GlobalWeatherRepository.csv here (not committed)
│   └── processed/            # Cleaned data written here by the pipeline
├── src/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_basic_forecasting.py
│   ├── 04_anomaly_detection.py       # advanced
│   ├── 05_multi_model_ensemble.py    # advanced
│   ├── 06_unique_analyses.py         # advanced
│   └── run_all.py
├── notebooks/                # Optional: combined narrative notebook for the write-up
├── visuals/                  # All generated charts/plots land here
├── reports/                  # CSV summaries + model comparison tables
├── requirements.txt
└── README.md
Setup
# 1. Clone your repo and cd into it
git clone <your-repo-url>
cd weather-forecasting-assessment

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place it at:
#    data/raw/GlobalWeatherRepository.csv
Running the pipeline
# Run everything end to end
python src/run_all.py

# Or run steps individually
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_basic_forecasting.py       # Basic assessment stops here
python src/04_anomaly_detection.py       # Advanced from here on
python src/05_multi_model_ensemble.py
python src/06_unique_analyses.py
All charts are saved to visuals/, and tabular results (model comparison, geographic summaries) to reports/.

Methodology
1. Data Cleaning (01_data_cleaning.py)
Parses last_updated into a proper datetime and derives year, month, day_of_week.
Missing numeric values are imputed with the country-level median (falling back to the global median); missing categoricals use the mode.
Outliers in temperature, precipitation, humidity, pressure, wind, and air-quality columns are winsorized using the IQR method rather than dropped, to preserve full weather records.
Key numeric columns are z-score normalized (*_z columns) for use in distance-based models.
2. Exploratory Data Analysis (02_eda.py)
Correlation heatmap across weather and air-quality features.
Global daily average temperature and precipitation trend lines.
Top hottest countries by average temperature.
Temperature vs humidity scatter to check the expected inverse-ish relationship.
3. Basic Forecasting (03_basic_forecasting.py)
Aggregates to a daily global average temperature series (the required last_updated time-series view).
Fits a SARIMA(2,1,2)(1,1,1,7) model (weekly seasonality) on all but the last 14 days.
Evaluates on held-out days with MAE, RMSE, and MAPE.
4. Advanced: Anomaly Detection (04_anomaly_detection.py)
Isolation Forest across multivariate weather features to flag unusual records.
Rolling z-score (30-day window) on the daily temperature series to flag anomalous days, visualized on the trend line.
5. Advanced: Multi-Model Forecasting + Ensemble (05_multi_model_ensemble.py)
Compares SARIMA, Prophet, and XGBoost (trained on lag/calendar features, forecast iteratively).
Builds a simple-average ensemble of all three and compares MAE/RMSE across all four in reports/model_comparison.csv.
6. Advanced: Unique Analyses (06_unique_analyses.py)
Climate analysis: monthly temperature patterns by continent.
Environmental impact: correlation between air-quality indices (PM2.5, PM10, CO, etc.) and weather parameters.
Feature importance: Random Forest built-in importance + permutation importance for predicting temperature.
Spatial analysis: scatter map of average temperature by latitude/longitude.
Geographical patterns: continent-level summary table of temperature, precipitation, and humidity.
Key Insights
(Fill in after running the pipeline on the actual dataset — e.g., strongest correlations found, best-performing forecast model and its error, most important predictive features, any notable regional/climate patterns or anomalies.)

Limitations
The dataset is a snapshot-style daily pull per city rather than a long historical record per location, so multi-year climate-trend claims should be made cautiously.
The forecasting models here are illustrative — for production-grade forecasting, city-level models with more historical depth would outperform the global aggregate approach used here.
How to Reproduce
See Setup and Running the pipeline above. Total runtime on the full dataset is a few minutes on a laptop CPU (XGBoost and Prophet steps are the slowest).
