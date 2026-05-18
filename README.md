# 🏭 IoT Anomaly Detection System

**DLBDSMTP01 – Project: From Model to Production**
*Task 1: Anomaly Detection in an IoT Setting (Stream Processing)*

---

## Overview

This project implements an **end-to-end anomaly detection system** for a wind turbine component factory. Factory sensors continuously measure **temperature**, **humidity**, and **sound volume** during the production cycle. The system ingests this stream of sensor data, predicts anomaly scores in real time via a REST API, and provides monitoring & alerting capabilities.

### Key Components

| Component | File | Purpose |
|---|---|---|
| Data Generator | `data_generator.py` | Creates synthetic sensor data for training |
| Model Training | `train_model.py` | Trains an Isolation Forest and saves artifacts |
| REST API | `app.py` | Serves predictions via HTTP endpoints |
| Stream Simulator | `stream_simulator.py` | Simulates real-time sensor data streams |
| Architecture | `architecture_diagram.html` | Visual system architecture diagram |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python data_generator.py
```

This creates `data/sensor_data.csv` with 5,000 sensor readings (5% anomalies).

### 3. Train the model

```bash
python train_model.py
```

Trains an Isolation Forest model and saves:
- `models/anomaly_model.pkl` — serialised model
- `models/scaler.pkl` — fitted StandardScaler
- `models/metrics.json` — evaluation metrics

### 4. Start the API

```bash
python app.py
```

The API starts on `http://localhost:5000`. Open the URL in a browser to see the **live monitoring dashboard**.

### 5. Simulate sensor streams

In a separate terminal:

```bash
python stream_simulator.py
```

Options:
```bash
python stream_simulator.py --interval 0.5      # 2 readings per second
python stream_simulator.py --anomaly-rate 0.1   # 10% anomalies
python stream_simulator.py --max-readings 100   # Stop after 100 readings
```

---

## API Reference

### `POST /predict` — Real-time prediction

**Request:**
```json
{
    "temperature": 72.5,
    "humidity": 45.0,
    "sound_volume": 82.3
}
```

**Response:**
```json
{
    "anomaly_score": 0.1234,
    "is_anomaly": false,
    "confidence": 0.87,
    "timestamp": "2025-01-15T10:30:00",
    "input": { ... }
}
```

### `POST /predict/batch` — Batch predictions

**Request:**
```json
{
    "readings": [
        {"temperature": 72.5, "humidity": 45.0, "sound_volume": 82.3},
        {"temperature": 95.1, "humidity": 78.2, "sound_volume": 112.0}
    ]
}
```

### `GET /health` — Health check
### `GET /metrics` — Monitoring metrics
### `GET /model/info` — Model metadata
### `GET /` — Live dashboard

---

## Architecture

Open `architecture_diagram.html` in a browser to view the full system architecture.

**Data flow:**
1. Factory sensors (temp, humidity, sound) emit readings continuously
2. Stream ingestion layer captures readings as JSON
3. Readings are sent via HTTP POST to the Flask REST API
4. API scales features using a pre-fitted StandardScaler
5. Isolation Forest model produces an anomaly score
6. Response (score, label, confidence) is returned to the caller
7. All predictions are logged and monitored via `/metrics` and the dashboard

---

## Monitoring

The system provides several monitoring capabilities:

- **`/metrics` endpoint** — exposes total predictions, anomaly count, anomaly rate, error count, and average latency
- **Live dashboard** (`/`) — auto-refreshing HTML page showing KPIs and recent predictions
- **Log files** — all API activity and anomaly alerts are logged to `logs/api.log`
- **Anomaly alerts** — anomaly detections are logged with `WARNING` level for easy filtering

---

## Model Details

- **Algorithm:** Isolation Forest (scikit-learn)
- **Features:** temperature (°C), humidity (%), sound_volume (dB)
- **Contamination:** 5% (matching expected anomaly rate)
- **Scaler:** StandardScaler (zero mean, unit variance)
- **Serialisation:** joblib (.pkl files)

The model is intentionally kept simple per the project requirements. The focus is on the production-ready architecture, not model complexity.

---

## Project Structure

```
iot-anomaly-detection/
├── app.py                      # Flask REST API
├── data_generator.py           # Synthetic data generation
├── train_model.py              # Model training pipeline
├── stream_simulator.py         # Real-time stream simulator
├── architecture_diagram.html   # System architecture visualisation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── sensor_data.csv         # Generated training data
├── models/
│   ├── anomaly_model.pkl       # Trained model
│   ├── scaler.pkl              # Fitted scaler
│   └── metrics.json            # Evaluation metrics
└── logs/
    └── api.log                 # API request logs
```

---

## Answers to Module Questions

**What were the challenges of integrating a predictive model into an application or service?**
Key challenges include serialisation/deserialisation of the model, ensuring feature scaling consistency between training and inference, managing model versioning, and handling malformed input gracefully.

**What are the constraints of implementing a predictive model as a service?**
Latency requirements for real-time predictions, memory footprint of the loaded model, concurrent request handling, and ensuring the API remains available under load.

**Which requirements for data acquisition, storage, and processing had to be met?**
Data must be ingested continuously (stream), validated for correct schema, scaled using the same parameters as training, and predictions must be returned in milliseconds.

**What are monitoring components required for reliable execution?**
Prediction latency tracking, error rate monitoring, anomaly rate trends (to detect model drift), health checks, structured logging, and a dashboard for human oversight.

---

*IU International University of Applied Sciences*
