"""Generate fictional equipment sensor data for a learning demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def main(n_rows=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    temperature = np.clip(rng.normal(70, 12, n_rows), 35, 115)
    vibration = np.clip(rng.gamma(2.2, 1.5, n_rows), 0.2, 18)
    pressure = np.clip(rng.normal(6.5, 1.1, n_rows), 2, 11)
    rpm = rng.integers(700, 3201, n_rows)
    load = np.clip(rng.normal(65, 18, n_rows), 10, 100)
    hours = rng.integers(100, 30001, n_rows)
    maintenance_age = rng.integers(1, 731, n_rows)

    # Illustrative formula used only to create a synthetic training target.
    score = ((temperature - 65) * 0.035 + vibration * 0.22
             + np.maximum(0, 5.5 - pressure) * 0.30
             + (load - 60) * 0.012 + (hours / 10000) * 0.35
             + (maintenance_age / 365) * 0.55)
    score += rng.normal(0, 0.75, n_rows)
    probability = 1 / (1 + np.exp(-(score - 3.4)))
    target = rng.binomial(1, probability)

    df = pd.DataFrame({
        "temperature_c": temperature.round(1),
        "vibration_mm_s": vibration.round(2),
        "pressure_bar": pressure.round(2),
        "rpm": rpm,
        "load_pct": load.round(1),
        "operating_hours": hours,
        "days_since_maintenance": maintenance_age,
        "failure_within_30_days": target,
    })
    path = DATA_DIR / "equipment_sensor_data.csv"
    df.to_csv(path, index=False)
    print(f"Created {len(df)} fictional observations: {path}")
    print(f"Synthetic event rate: {df['failure_within_30_days'].mean():.1%}")

if __name__ == "__main__":
    main()
