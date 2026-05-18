"""
app.py
------
RESTful API for the IoT Anomaly Detection Service.

Endpoints:
  POST /predict          – Single prediction (real-time stream)
  POST /predict/batch    – Batch predictions
  GET  /health           – Health check
  GET  /metrics          – Prometheus-style metrics
  GET  /model/info       – Model metadata & performance metrics
  GET  /                 – Dashboard (HTML)

The API accepts JSON sensor readings and returns anomaly scores.
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import json
import os
import time
import logging
from datetime import datetime
from functools import wraps
from collections import deque

# ── Configuration ─────────────────────────────────────────────────
MODEL_PATH = "models/anomaly_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"
LOG_DIR = "logs"
FEATURES = ["temperature", "humidity", "sound_volume"]

# ── Logging setup ─────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────
logger.info("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METRICS_PATH, "r") as f:
    model_metrics = json.load(f)
logger.info("Model loaded successfully.")

# ── Monitoring state ──────────────────────────────────────────────
monitoring = {
    "total_requests": 0,
    "total_predictions": 0,
    "anomalies_detected": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "start_time": datetime.now().isoformat(),
    "recent_predictions": deque(maxlen=100),  # rolling window
    "latencies": deque(maxlen=1000),
}


def track_request(f):
    """Decorator to track request metrics."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        monitoring["total_requests"] += 1
        start = time.time()
        try:
            result = f(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            monitoring["latencies"].append(latency_ms)
            monitoring["avg_latency_ms"] = round(
                np.mean(list(monitoring["latencies"])), 2
            )
            return result
        except Exception as e:
            monitoring["errors"] += 1
            logger.error(f"Request error: {e}")
            raise
    return wrapper


def validate_input(data: dict) -> tuple:
    """Validate that all required features are present and numeric."""
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return False, f"Missing features: {missing}"
    for f in FEATURES:
        if not isinstance(data[f], (int, float)):
            return False, f"Feature '{f}' must be numeric, got {type(data[f]).__name__}"
    return True, ""


# ── ENDPOINTS ─────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
@track_request
def predict():
    """
    Single prediction endpoint for real-time stream processing.

    Request Body (JSON):
        {
            "temperature": 72.5,
            "humidity": 45.0,
            "sound_volume": 82.3
        }

    Response (JSON):
        {
            "anomaly_score": -0.123,
            "is_anomaly": false,
            "confidence": 0.87,
            "timestamp": "2025-01-15T10:30:00"
        }
    """
    data = request.get_json(force=True)

    valid, error_msg = validate_input(data)
    if not valid:
        return jsonify({"error": error_msg}), 400

    # Extract features and scale
    features = np.array([[data[f] for f in FEATURES]])
    features_scaled = scaler.transform(features)

    # Predict
    raw_score = model.decision_function(features_scaled)[0]
    prediction = model.predict(features_scaled)[0]
    is_anomaly = bool(prediction == -1)

    # Confidence: normalize score to [0, 1] range
    # More negative = more anomalous
    confidence = round(float(1 / (1 + np.exp(-raw_score * 5))), 4)

    result = {
        "anomaly_score": round(float(raw_score), 6),
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "input": data,
    }

    # Update monitoring
    monitoring["total_predictions"] += 1
    if is_anomaly:
        monitoring["anomalies_detected"] += 1
        logger.warning(f"ANOMALY DETECTED | score={raw_score:.4f} | input={data}")
    monitoring["recent_predictions"].append(result)

    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
@track_request
def predict_batch():
    """
    Batch prediction endpoint.

    Request Body (JSON):
        {
            "readings": [
                {"temperature": 72.5, "humidity": 45.0, "sound_volume": 82.3},
                {"temperature": 95.1, "humidity": 78.2, "sound_volume": 112.0}
            ]
        }
    """
    data = request.get_json(force=True)
    readings = data.get("readings", [])

    if not readings:
        return jsonify({"error": "No readings provided"}), 400

    results = []
    for i, reading in enumerate(readings):
        valid, error_msg = validate_input(reading)
        if not valid:
            results.append({"index": i, "error": error_msg})
            continue

        features = np.array([[reading[f] for f in FEATURES]])
        features_scaled = scaler.transform(features)
        raw_score = model.decision_function(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        is_anomaly = bool(prediction == -1)
        confidence = round(float(1 / (1 + np.exp(-raw_score * 5))), 4)

        results.append({
            "index": i,
            "anomaly_score": round(float(raw_score), 6),
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "input": reading,
        })

        monitoring["total_predictions"] += 1
        if is_anomaly:
            monitoring["anomalies_detected"] += 1

    anomaly_count = sum(1 for r in results if r.get("is_anomaly", False))
    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "anomalies": anomaly_count,
            "normal": len(results) - anomaly_count,
            "timestamp": datetime.now().isoformat(),
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime_since": monitoring["start_time"],
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus-compatible metrics endpoint."""
    anomaly_rate = (
        monitoring["anomalies_detected"] / monitoring["total_predictions"]
        if monitoring["total_predictions"] > 0 else 0
    )
    return jsonify({
        "total_requests": monitoring["total_requests"],
        "total_predictions": monitoring["total_predictions"],
        "anomalies_detected": monitoring["anomalies_detected"],
        "anomaly_rate": round(anomaly_rate, 4),
        "errors": monitoring["errors"],
        "avg_latency_ms": monitoring["avg_latency_ms"],
        "uptime_since": monitoring["start_time"],
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and training metrics."""
    return jsonify(model_metrics)


@app.route("/", methods=["GET"])
def dashboard():
    """Simple monitoring dashboard."""
    anomaly_rate = (
        monitoring["anomalies_detected"] / monitoring["total_predictions"] * 100
        if monitoring["total_predictions"] > 0 else 0
    )
    recent = list(monitoring["recent_predictions"])[-10:]
    recent.reverse()

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IoT Anomaly Detection – Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
            h1 { font-size: 1.8rem; margin-bottom: 1.5rem; color: #38bdf8; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
            .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
            .card .label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
            .card .value { font-size: 2rem; font-weight: 700; margin-top: 0.5rem; }
            .anomaly { color: #f87171; }
            .normal { color: #4ade80; }
            .info { color: #38bdf8; }
            table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
            th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }
            th { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }
            .badge { padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
            .badge.red { background: #991b1b; color: #fca5a5; }
            .badge.green { background: #166534; color: #86efac; }
            h2 { font-size: 1.3rem; margin-top: 2rem; margin-bottom: 0.5rem; color: #cbd5e1; }
        </style>
    </head>
    <body>
        <h1>IoT Anomaly Detection Dashboard</h1>
        <div class="grid">
            <div class="card">
                <div class="label">Total Predictions</div>
                <div class="value info">{{ total_predictions }}</div>
            </div>
            <div class="card">
                <div class="label">Anomalies Detected</div>
                <div class="value anomaly">{{ anomalies_detected }}</div>
            </div>
            <div class="card">
                <div class="label">Anomaly Rate</div>
                <div class="value {% if anomaly_rate > 10 %}anomaly{% else %}normal{% endif %}">{{ anomaly_rate }}%</div>
            </div>
            <div class="card">
                <div class="label">Avg Latency</div>
                <div class="value info">{{ avg_latency }}ms</div>
            </div>
            <div class="card">
                <div class="label">Errors</div>
                <div class="value {% if errors > 0 %}anomaly{% else %}normal{% endif %}">{{ errors }}</div>
            </div>
        </div>
        <h2>Recent Predictions</h2>
        <table>
            <tr><th>Time</th><th>Temp (°C)</th><th>Humidity (%)</th><th>Sound (dB)</th><th>Score</th><th>Status</th></tr>
            {% for p in recent %}
            <tr>
                <td>{{ p.timestamp[:19] }}</td>
                <td>{{ p.input.temperature }}</td>
                <td>{{ p.input.humidity }}</td>
                <td>{{ p.input.sound_volume }}</td>
                <td>{{ p.anomaly_score }}</td>
                <td><span class="badge {% if p.is_anomaly %}red{% else %}green{% endif %}">
                    {% if p.is_anomaly %}ANOMALY{% else %}NORMAL{% endif %}
                </span></td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    return render_template_string(
        html,
        total_predictions=monitoring["total_predictions"],
        anomalies_detected=monitoring["anomalies_detected"],
        anomaly_rate=round(anomaly_rate, 2),
        avg_latency=monitoring["avg_latency_ms"],
        errors=monitoring["errors"],
        recent=recent,
    )


if __name__ == "__main__":
    logger.info("Starting IoT Anomaly Detection API on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
