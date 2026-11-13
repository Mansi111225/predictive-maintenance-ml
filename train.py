"""Train and evaluate a Random Forest predictive-maintenance demo."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "equipment_sensor_data.csv"
FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm",
            "load_pct", "operating_hours", "days_since_maintenance"]
TARGET = "failure_within_30_days"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python generate_data.py` first.")
    df = pd.read_csv(DATA)
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found; clean the dataset before training.")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    preprocess = ColumnTransformer([("numeric", StandardScaler(), FEATURES)])
    model = Pipeline([
        ("preprocessing", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=9, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test, pred, target_names=["No simulated event", "Simulated event"],
        zero_division=0
    )
    auc = roc_auc_score(y_test, prob)

    model_dir, report_dir = ROOT / "models", ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "predictive_maintenance_pipeline.joblib")
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No event", "Event"], cmap="Blues"
    )
    plt.title("Confusion Matrix – Synthetic Data")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    (report_dir / "metrics.txt").write_text(
        "PREDICTIVE MAINTENANCE – SYNTHETIC DEMO\\n"
        f"Test rows: {len(y_test)}\\nROC-AUC: {auc:.3f}\\n\\n{report}\\n"
        "Metrics describe only synthetic, rule-generated data.\\n",
        encoding="utf-8"
    )
    print("Model saved:", model_dir / "predictive_maintenance_pipeline.joblib")
    print("ROC-AUC:", round(auc, 3))
    print("\\nClassification report:\\n", report)
    print("Reports saved in:", report_dir)

if __name__ == "__main__":
    main()
