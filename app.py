"""Streamlit UI for the synthetic predictive-maintenance model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "predictive_maintenance_pipeline.joblib"

st.set_page_config(page_title="Predictive Maintenance Demo", page_icon="⚙️")
st.title("Predictive Maintenance – Equipment Health Demo")
st.caption("Educational prototype using synthetic data. Not for real maintenance or safety decisions.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python generate_data.py` and `python train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.subheader("Enter equipment indicators")
with st.form("maintenance_form"):
    temperature = st.slider("Operating temperature (°C)", 35.0, 115.0, 70.0, 0.5)
    vibration = st.slider("Vibration (mm/s)", 0.2, 18.0, 3.0, 0.1)
    pressure = st.slider("Operating pressure (bar)", 2.0, 11.0, 6.5, 0.1)
    rpm = st.slider("Rotational speed (RPM)", 700, 3200, 1800, 50)
    load = st.slider("Operating load (%)", 10.0, 100.0, 65.0, 1.0)
    hours = st.number_input("Cumulative operating hours", 100, 30000, 8000, 100)
    age = st.number_input("Days since last maintenance", 1, 730, 90, 1)
    submitted = st.form_submit_button("Estimate failure risk")

if submitted:
    sample = pd.DataFrame([{
        "temperature_c": temperature, "vibration_mm_s": vibration,
        "pressure_bar": pressure, "rpm": rpm, "load_pct": load,
        "operating_hours": hours, "days_since_maintenance": age
    }])
    probability = float(model.predict_proba(sample)[0, 1])
    prediction = int(model.predict(sample)[0])
    st.divider()
    st.subheader("Demo model output")
    st.metric("Estimated probability of synthetic failure class", f"{probability:.1%}")
    if prediction:
        st.warning("Model classification: Higher estimated risk")
    else:
        st.success("Model classification: Lower estimated risk")
    st.progress(probability)
    st.info("This is a model trained on fictional, rule-generated data. It is not a real equipment-health assessment or maintenance recommendation.")
