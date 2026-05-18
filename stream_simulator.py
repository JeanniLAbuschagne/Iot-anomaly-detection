"""
stream_simulator.py
-------------------
Simulates a continuous stream of IoT sensor data hitting the prediction API.
Mimics factory sensors sending readings every few seconds.

Usage:
    python stream_simulator.py                     # default: 1 reading/sec
    python stream_simulator.py --interval 0.5      # 2 readings/sec
    python stream_simulator.py --anomaly-rate 0.1  # 10% anomalies
"""

import requests
import numpy as np
import time
import argparse
import sys
from datetime import datetime


API_URL = "http://localhost:5000/predict"

# Normal operating ranges
NORMAL_PARAMS = {
    "temperature":  {"mean": 70, "std": 4},
    "humidity":     {"mean": 40, "std": 5},
    "sound_volume": {"mean": 80, "std": 4},
}

# Anomalous ranges
ANOMALY_PARAMS = {
    "temperature":  {"mean": 95, "std": 6},
    "humidity":     {"mean": 75, "std": 6},
    "sound_volume": {"mean": 110, "std": 5},
}


def generate_reading(rng: np.random.Generator, is_anomaly: bool = False) -> dict:
    """Generate a single sensor reading."""
    params = ANOMALY_PARAMS if is_anomaly else NORMAL_PARAMS
    return {
        "temperature":  round(float(rng.normal(params["temperature"]["mean"], params["temperature"]["std"])), 2),
        "humidity":     round(float(rng.normal(params["humidity"]["mean"], params["humidity"]["std"])), 2),
        "sound_volume": round(float(rng.normal(params["sound_volume"]["mean"], params["sound_volume"]["std"])), 2),
    }


def run_stream(interval: float = 1.0, anomaly_rate: float = 0.05, max_readings: int = None):
    """
    Send continuous sensor readings to the prediction API.

    Parameters
    ----------
    interval : float
        Seconds between readings.
    anomaly_rate : float
        Probability of generating an anomalous reading.
    max_readings : int or None
        Stop after this many readings (None = run forever).
    """
    rng = np.random.default_rng()
    count = 0
    anomalies = 0
    errors = 0

    print("=" * 65)
    print("  IoT SENSOR STREAM SIMULATOR")
    print(f"  Target API : {API_URL}")
    print(f"  Interval   : {interval}s")
    print(f"  Anomaly %  : {anomaly_rate * 100:.0f}%")
    print("=" * 65)
    print()

    try:
        while max_readings is None or count < max_readings:
            # Decide if this reading is anomalous
            is_anomaly = rng.random() < anomaly_rate
            reading = generate_reading(rng, is_anomaly)

            try:
                response = requests.post(API_URL, json=reading, timeout=5)
                result = response.json()
                count += 1

                detected = result.get("is_anomaly", False)
                score = result.get("anomaly_score", 0)

                if detected:
                    anomalies += 1
                    icon = "[ANOMALY]"
                    status = "ANOMALY"
                else:
                    icon = "[OK]"
                    status = "NORMAL "

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(
                    f"  {icon} [{timestamp}] #{count:04d}  "
                    f"T={reading['temperature']:6.1f}°C  "
                    f"H={reading['humidity']:5.1f}%  "
                    f"S={reading['sound_volume']:5.1f}dB  "
                    f"--> {status}  (score: {score:+.4f})"
                )

            except requests.exceptions.ConnectionError:
                errors += 1
                print(f"  Connection error - is the API running on {API_URL}?")
            except Exception as e:
                errors += 1
                print(f"  Error: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 65)
    print(f"  Stream finished.")
    print(f"  Total readings : {count}")
    print(f"  Anomalies      : {anomalies}")
    print(f"  Errors         : {errors}")
    print(f"  Anomaly rate   : {anomalies/count*100:.1f}%" if count > 0 else "")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="IoT Sensor Stream Simulator")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="Anomaly probability (0-1)")
    parser.add_argument("--max-readings", type=int, default=None, help="Max readings (default: unlimited)")
    args = parser.parse_args()

    run_stream(
        interval=args.interval,
        anomaly_rate=args.anomaly_rate,
        max_readings=args.max_readings,
    )


if __name__ == "__main__":
    main()
