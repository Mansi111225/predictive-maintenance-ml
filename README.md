# Predictive Maintenance – Synthetic Equipment Demo

A beginner-friendly machine-learning project that estimates whether a fictional equipment unit may experience a failure event within the next 30 days using simulated sensor and maintenance indicators.

> **Important:** All data is synthetic and generated for learning. This is not trained on Volvo data, real vehicle telemetry, or verified failure records. It must not be used for real maintenance or safety decisions.

## What it demonstrates
- Python and Pandas data generation
- Scikit-learn preprocessing and Random Forest classification
- Train/test split and model evaluation
- Precision, recall, F1-score, ROC-AUC and confusion matrix
- Saving a trained model and serving predictions with Streamlit

## Structure
```text
predictive-maintenance/
├── app.py
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
└── reports/
```

## Setup (Python 3.10+)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Generate synthetic data
```powershell
python generate_data.py
```
Creates 2,000 fictional equipment observations. The target is generated from illustrative rules plus randomness; it is not a real failure label.

## Train and evaluate
```powershell
python train.py
```
Creates `models/predictive_maintenance_pipeline.joblib`, `reports/metrics.txt` and `reports/confusion_matrix.png`.

High scores only show that the model learned patterns in the simulated dataset. They do not demonstrate real-world predictive performance.

## Run the app
```powershell
python -m streamlit run app.py
```

## Features
| Feature | Meaning |
|---|---|
| temperature_c | Simulated operating temperature (°C) |
| vibration_mm_s | Simulated vibration velocity |
| pressure_bar | Simulated operating pressure |
| rpm | Simulated rotational speed |
| load_pct | Simulated operating load (%) |
| operating_hours | Cumulative operating hours |
| days_since_maintenance | Days since last maintenance |
| failure_within_30_days | Synthetic target: 1 = simulated event, 0 = no event |

These are generic fictional indicators, not specifications or thresholds for any Volvo product.

## Limitations and responsible use
- All records and labels are synthetic.
- Real deployment requires reliable sensor histories, verified failure labels, domain-approved features, time-aware validation, leakage checks, calibration, explainability, safety review and ongoing monitoring.
- Model output is not a substitute for maintenance procedures, fault diagnosis or engineering judgement.

## GitHub publishing
Create an empty public repository named `predictive-maintenance`, then run:
```powershell
git init
git add .
git commit -m "Add predictive maintenance ML demo"
git branch -M main
git remote add origin https://github.com/Mansi111225/predictive-maintenance.git
git push -u origin main
```
Never publish confidential company data, credentials or private files.
