"""
train_model.py
--------------
Trains an Isolation Forest model on the synthetic sensor data.
Evaluates performance and serialises the model for serving.

The Isolation Forest is ideal here because:
  - It works well with numerical features
  - It does not require labelled anomaly data (unsupervised)
  - It scales to streaming / real-time inference
  - It is lightweight and fast at prediction time
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime


FEATURES = ["temperature", "humidity", "sound_volume"]
DATA_PATH = "data/sensor_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def train():
    # ── 1. Load data ──────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} rows  |  Anomalies: {df['is_anomaly'].sum():.0f} "
          f"({df['is_anomaly'].mean()*100:.1f}%)")

    X = df[FEATURES].values
    y_true = df["is_anomaly"].values

    # ── 2. Train / Test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=42, stratify=y_true
    )

    # ── 3. Feature scaling ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. Train Isolation Forest ─────────────────────────────────
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # matches our anomaly ratio
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    # ── 5. Evaluate ───────────────────────────────────────────────
    # Isolation Forest returns -1 for anomalies, 1 for normal
    y_pred_raw = model.predict(X_test_scaled)
    y_pred = (y_pred_raw == -1).astype(int)  # convert to 0/1

    # Anomaly scores (lower = more anomalous)
    scores = model.decision_function(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": cm.tolist(),
        "features": FEATURES,
        "model_type": "IsolationForest",
        "contamination": 0.05,
        "n_estimators": 200,
    }

    # ── 6. Save model, scaler, metrics ────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved --> {MODEL_PATH}")
    print(f"Scaler saved --> {SCALER_PATH}")
    print(f"Metrics saved --> {METRICS_PATH}")
    print(f"\n   F1 Score: {metrics['f1_score']}")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall: {metrics['recall']}")


if __name__ == "__main__":
    train()
