"""
data_generator.py
-----------------
Generates synthetic IoT sensor data for a wind turbine component factory.
Features: temperature (°C), humidity (%), sound_volume (dB)
Labels:   0 = normal, 1 = anomaly

Normal operating ranges (based on domain expert consultation):
  - Temperature:  60–80 °C
  - Humidity:     30–50 %
  - Sound Volume: 70–90 dB

Anomalous readings deviate significantly from these ranges.
"""

import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta


def generate_sensor_data(n_samples: int = 5000, anomaly_ratio: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic sensor data with a controllable anomaly ratio.

    Parameters
    ----------
    n_samples : int
        Total number of data points to generate.
    anomaly_ratio : float
        Proportion of anomalous samples (0.0 – 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, temperature, humidity, sound_volume, is_anomaly
    """
    rng = np.random.default_rng(seed)

    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # --- Normal data ---
    temp_normal = rng.normal(loc=70, scale=4, size=n_normal)       # °C
    hum_normal = rng.normal(loc=40, scale=5, size=n_normal)        # %
    sound_normal = rng.normal(loc=80, scale=4, size=n_normal)      # dB

    # --- Anomalous data (values outside typical operating ranges) ---
    temp_anomaly = rng.choice(
        [rng.normal(loc=95, scale=5, size=n_anomalies),   # overheating
         rng.normal(loc=40, scale=5, size=n_anomalies)],  # undercooling
    )
    hum_anomaly = rng.choice(
        [rng.normal(loc=75, scale=5, size=n_anomalies),   # too humid
         rng.normal(loc=10, scale=3, size=n_anomalies)],  # too dry
    )
    sound_anomaly = rng.choice(
        [rng.normal(loc=110, scale=5, size=n_anomalies),  # loud grinding
         rng.normal(loc=50, scale=5, size=n_anomalies)],  # unusually quiet
    )

    # Combine
    temperature = np.concatenate([temp_normal, temp_anomaly])
    humidity = np.concatenate([hum_normal, hum_anomaly])
    sound_volume = np.concatenate([sound_normal, sound_anomaly])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])

    # Generate timestamps (one reading every 10 seconds)
    start_time = datetime(2025, 1, 1, 8, 0, 0)
    timestamps = [start_time + timedelta(seconds=10 * i) for i in range(n_samples)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "sound_volume": np.round(sound_volume, 2),
        "is_anomaly": labels.astype(int),
    })

    # Shuffle so anomalies are distributed throughout
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Re-assign sequential timestamps after shuffle
    df["timestamp"] = timestamps

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IoT sensor data")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--anomaly_ratio", type=float, default=0.05, help="Anomaly ratio (0-1)")
    parser.add_argument("--output", type=str, default="data/sensor_data.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_sensor_data(n_samples=args.n_samples, anomaly_ratio=args.anomaly_ratio)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} samples ({df['is_anomaly'].sum():.0f} anomalies) --> {args.output}")
    print(f"\nData Summary:\n{df.describe().round(2)}")


if __name__ == "__main__":
    main()
