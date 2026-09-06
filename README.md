# SIH 26025 — AI-Enabled Real-Time Mine Subsidence Monitoring, Prediction and Early Warning System

> **Smart India Hackathon 2026** | Problem Code: **PS_025**  
> **Domain**: Mining, Geotechnical Engineering & Disaster Mitigation  
> **Objective**: Low-cost, offline-resilient underground coal mine strata subsidence monitoring, Time-to-Failure (TTF) collapse prediction, and DGMS SCAMP compliance alerting.

---

## 🏗️ System Architecture

```text
               +----------------------------------------+
               | ESP32 / Tilt / Displacement / LoRa Nodes|
               +----------------------------------------+
                                   │
                                   ▼
                  +─────────────────────────────────+
                  |  FastAPI Backend (Port: 8000)   |
                  |  - /telemetry (Ingest)          |
                  |  - /ws/telemetry (WebSocket)    |
                  +─────────────────────────────────+
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
   +─────────────────────────────+    +───────────────────────────+
   |  30-Reading Rolling Window  |    | Offline Local SQLite DB   |
   |  - Per-Node Buffer Isolation|    | - Telemetry & Audit Logs  |
   +─────────────────────────────+    +───────────────────────────+
                  │
                  ▼
   +─────────────────────────────+
   |   Member 1 LSTM Model       |
   |   - Input: [v, def, 1/v]    |
   |   - Output: TTF in Hours    |
   +─────────────────────────────+
                  │
                  ▼
   +──────────────────────────────────────────────────────────────+
   |  Multi-Factor Sensor Fusion & Geotechnical Risk Engine       |
   |  - Temporal Trend Analysis (STABLE -> ACCELERATION -> CRIT)  |
   |  - Spatial Neighborhood Correlation Graph (3x3 Matrix)       |
   |  - Isolated Sensor Fault & Anomaly Diagnostics Engine        |
   +──────────────────────────────────────────────────────────────+
                  │
                  ▼
   +──────────────────────────────────────────────────────────────+
   |  Streamlit Real-Time Dashboard (Port: 8501)                  |
   |  - Dual Kinematic Risk Charts (Tilt vs TTF, Velocity)        |
   |  - Interactive 3x3 Gallery Grid & Spatial Risk Map           |
   |  - DGMS SCAMP Compliance Export & Scenario Simulator Control |
   +──────────────────────────────────────────────────────────────+
```

---

## 📂 Project Structure

```text
SIH2026/
├── ml/                                 # Member 1 — Machine Learning Models & Training
│   ├── ttf_lstm_model.keras            # Trained Sequential LSTM model
│   ├── lstm_feature_scaler.pkl         # StandardScaler (velocity, deformation, 1/v)
│   ├── lstm_target_scaler.pkl          # Target StandardScaler for TTF (hours)
│   ├── predict_ttf.py                  # Import-safe inference function
│   ├── risk_engine.py                  # Base TTF risk classification
│   └── all_creep_scenarios.csv         # Synthetic rock creep training dataset
│
├── backend/                            # Backend & Sensor Fusion Integration
│   ├── main.py                         # FastAPI app, CORS, WebSocket, Lifespan
│   ├── api/
│   │   ├── telemetry.py                # POST /telemetry, GET /latest, GET /telemetry/history
│   │   ├── prediction.py               # GET /prediction/{node_id}, GET /risk, GET /risk/map
│   │   ├── nodes.py                    # GET /nodes, GET /nodes/{node_id}, GET /nodes/{node_id}/health
│   │   ├── alerts.py                   # GET /alerts, GET /events
│   │   └── simulation.py               # POST /simulation/scenario, /start, /stop
│   ├── core/
│   │   ├── config.py                   # Thresholds, model paths, 3x3 grid coordinates
│   │   ├── conversions.py              # Telemetry-to-ML feature conversion
│   │   ├── risk_engine.py              # Multi-factor geotechnical fusion risk engine
│   │   ├── spatial.py                  # Spatial graph and neighbor correlation
│   │   └── health.py                   # Sensor fault isolation and heartbeat
│   ├── services/
│   │   ├── history_service.py          # Per-node 30-reading isolated rolling window
│   │   ├── prediction_service.py       # ML bridge with graceful fallback
│   │   └── alert_service.py            # Alert lifecycle management
│   ├── models/
│   │   └── schemas.py                  # Pydantic data schemas & contracts
│   └── storage/
│       └── database.py                 # SQLite offline-first local database
│
├── simulator/                          # Geotechnical Mine Simulator
│   └── mine_simulator.py               # Multi-node multi-scenario generator
│
├── frontend/                           # Member 3 — Streamlit Dashboard
│   ├── app.py                          # Real-time monitoring UI
│   └── requirements.txt
│
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10 or 3.11 installed.
* PowerShell or Terminal.

### 2. Environment Setup
```powershell
# In project root: c:\Users\sibad\Downloads\SIH2026
pip install -r requirements.txt
```

### 3. Running the Components

Open **two** PowerShell terminals:

#### Terminal 1: Start FastAPI Backend
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* API Documentation: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

#### Terminal 2: Start Streamlit Dashboard
```powershell
cd frontend
streamlit run app.py
```
* Dashboard URL: `http://localhost:8501`

---

## 🕹️ Interactive Geotechnical Simulator

In the dashboard sidebar, you can switch between 5 geotechnical and operational scenarios:

1. **`NORMAL`**: Stable background strata. Readings hover within standard creep limits.
2. **`RISING_DEFORMATION`**: Secondary creep progression. Elevated velocity across the Gallery 2 cluster.
3. **`ACCELERATING_DEFORMATION`**: Tertiary creep exponential acceleration. Velocity surges, TTF drops below 1.0 hour, and a **CRITICAL EVACUATION ALARM** is triggered.
4. **`SENSOR_FAULT`**: Physical hardware disturbance on an isolated node (`NODE_07`). The spatial engine recognizes that surrounding nodes (`NODE_04`, `NODE_08`, etc.) are stable, diagnosing an isolated hardware fault instead of triggering a false mine-wide evacuation.
5. **`COMMUNICATION_FAILURE`**: Packet loss / power failure on `NODE_09`. The health engine marks the node as `DEGRADED`/`OFFLINE`.

---

## 📡 API Contract Reference

### POST `/telemetry`
Ingest raw sensor payload:
```json
{
  "node_id": "NODE_05",
  "timestamp": "2026-09-06T15:00:00",
  "tilt_x": 1.45,
  "tilt_y": 0.82,
  "acceleration": 0.04,
  "displacement_mm": 14.2,
  "vibration": 0.05,
  "battery": 94.0,
  "signal_strength": -68.0
}
```

### GET `/latest`
Returns current sensor state, predicted TTF, and risk assessment:
```json
{
  "timestamp": "2026-09-06T15:00:00",
  "node_id": "NODE_05",
  "tilt_deg": 1.66,
  "ttf_hours": 2.45,
  "velocity_mm_h": 0.125,
  "deformation_mm": 14.2,
  "risk_status": "HIGH RISK",
  "risk_score": 62.4,
  "confidence": 0.94,
  "trend": "ACCELERATION DETECTED",
  "buffer_status": "30/30"
}
```

---

## 🛡️ Offline-First & DGMS SCAMP Compliance

1. **Offline Edge Computing**: Zero reliance on external cloud APIs. All LSTM inference, SQLite persistence, and spatial risk calculations execute locally on edge gateways.
2. **SCAMP Report Export**: One-click download of Strata Control and Monitoring Plan audit reports formatted in accordance with DGMS circular standards.
