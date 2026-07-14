"""
run_all.py
----------
Convenience entry point: runs the full pipeline end to end.
Comment out the advanced steps if you're only doing the Basic Assessment.

Run:
    python src/run_all.py
"""

import subprocess
import sys

STEPS = [
    "src/01_data_cleaning.py",
    "src/02_eda.py",
    "src/03_basic_forecasting.py",
    # --- Advanced assessment steps (optional) ---
    "src/04_anomaly_detection.py",
    "src/05_multi_model_ensemble.py",
    "src/06_unique_analyses.py",
]

for step in STEPS:
    print(f"\n{'='*60}\nRunning {step}\n{'='*60}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"Step {step} failed — stopping.")
        break
