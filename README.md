# ⚡ STRATA-X — AI-Driven Real-Time Mine Subsidence Intelligence & Early Warning System

<div align="center">

[![Smart India Hackathon 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-blue.svg?style=for-the-badge&logo=target)](https://www.sih.gov.in/)
[![Problem Code](https://img.shields.io/badge/Problem%20Code-PS__025-ff6f00.svg?style=for-the-badge)](https://www.sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![DGMS Compliant](https://img.shields.io/badge/DGMS-SCAMP%20Standard-2e7d32.svg?style=for-the-badge)](https://www.dgms.gov.in/)

**Low-cost, offline-resilient underground coal mine strata subsidence monitoring, Time-to-Failure (TTF) collapse forecasting, and DGMS SCAMP-compliant early warning system.**

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Geotechnical Theory](#-geotechnical-kinematics--fukuzono-law) • [Quickstart](#-quickstart-guide) • [API Reference](#-api-endpoints--contracts) • [SCAMP Standards](#-dgms-scamp-compliance-matrix)

</div>

---

## 📌 Executive Summary

Underground coal mining faces severe safety hazards from sudden roof falls, strata collapses, and gallery subsidence. **STRATA-X** is an end-to-end geotechnical intelligence system combining low-cost edge telemetry, deep learning kinematics (Fukuzono Inverse Velocity), spatial neighborhood graph correlation, and an offline-first architecture designed specifically for harsh subterranean environments with intermittent connectivity.

### 🌟 Key Value Propositions
* **🧠 Deep Learning TTF Prediction**: Sequential LSTM model trained on triaxial rock creep curves, predicting **Time-to-Failure (TTF in hours)** from multi-parameter rolling kinematics ($v$, $\Delta$, $1/v$, $\alpha$).
* **🌐 Spatial Neighborhood Correlation**: 3×3 gallery mesh graph that cross-validates localized sensor anomalies against neighboring nodes, preventing costly false mine-wide evacuation alarms.
* **⚡ 100% Offline-Resilient**: Runs entirely on edge gateways with zero external cloud dependencies using local SQLite WAL persistence and quantized model inference.
* **📋 DGMS SCAMP Automated Audits**: Direct one-click compliance export adhering to Directorate General of Mines Safety (DGMS) circulars and Strata Control and Monitoring Plan (SCAMP) thresholds.
* **🕹️ Real-Time Kinematic Simulator**: Integrated 5-scenario geotechnical simulator for training, disaster drill rehearsals, and fault-injection testing.

---

## 🏗️ System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │              Subterranean Strata Monitoring Nodes                      │
 │    [Node 1: ESP32 + Tilt + LVDT] ... [Node 9: LoRa 868MHz Mesh]        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ LoRa / Serial / HTTP Ingestion
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 STRATA-X Gateway Backend (FastAPI :8000)               │
 │                                                                        │
 │  ┌─────────────────────────┐            ┌───────────────────────────┐  │
 │  │ POST /telemetry         │            │ Per-Node 30-Reading       │  │
 │  │ GET /latest             │ ─────────► │ Isolated Rolling Buffer   │  │
 │  │ WebSocket /ws/telemetry │            │ (Velocity, Accel, 1/v)    │  │
 │  └─────────────────────────┘            └─────────────┬─────────────┘  │
 │                 │                                     │                │
 │                 ▼                                     ▼                │
 │  ┌─────────────────────────┐            ┌───────────────────────────┐  │
 │  │ Offline Local SQLite DB │            │ Sequential LSTM TTF Model │  │
 │  │ - Telemetry & Audit Log │            │ - Input: [v, def, 1/v]    │  │
 │  │ - Event Sequence Journal│            │ - Output: TTF (Hours)     │  │
 │  └─────────────────────────┘            └─────────────┬─────────────┘  │
 │                                                       │                │
 │                                                       ▼                │
 │  ┌──────────────────────────────────────────────────────────────────┐  │
 │  │    Multi-Factor Geotechnical Sensor Fusion & Risk Engine         │  │
 │  │    - Kinematic Phase Classifier (Primary / Secondary / Tertiary) │  │
 │  │    - 3×3 Gallery Spatial Correlation Matrix                      │  │
 │  │    - Node Diagnostic Heartbeat & Sensor Fault Isolation          │  │
 │  └──────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ WebSocket / REST Stream
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │            Mine Safety Command Dashboard (Streamlit :8501)             │
 │  - Real-Time Dual Kinematic Charts (Velocity, Deformation, TTF)        │
 │  - Interactive 3×3 Subterranean Spatial Heatmap                        │
 │  - Automated DGMS SCAMP Risk Matrix & PDF/CSV Incident Logs            │
 │  - Geotechnical Scenario Simulator Switchboard                         │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Geotechnical Kinematics & Fukuzono Law

STRATA-X incorporates the empirical **Fukuzono Inverse-Velocity Law** ($1985$) and Saito's creep curve formulations for progressive rock mass failure:

$$\frac{d\delta}{dt} = v(t) \quad \implies \quad \frac{1}{v(t)} \to 0 \quad \text{as } t \to t_f$$

```text
Deformation Rate (v)
      ▲
      │                                    Tertiary Creep (Exponential Acceleration)
      │                                       / 💥 Collapse at t_f
      │                     Secondary Creep  /
      │                    (Constant Rate)  /
      │               ┌────────────────────┘
      │  Primary     /
      │  (Decaying) /
      │  ──────────┘
      └───────────────────────────────────────────────────────► Time (t)
```

1. **Primary Creep (Stable)**: Transient deceleration after excavation adjustments.
2. **Secondary Creep (Caution)**: Steady-state slow plastic deformation.
3. **Tertiary Creep (Critical)**: Unstable acceleration where inverse velocity $\frac{1}{v}$ drops linearly toward zero.

---

## 📂 Repository Structure

```text
STRATA-X/
├── backend/                            # FastAPI Gateway & Sensor Fusion Backend
│   ├── main.py                         # Application lifecycle, CORS, WebSocket, Router mounts
│   ├── api/                            # REST & WebSocket endpoints
│   │   ├── telemetry.py                # Telemetry ingestion & historical queries
│   │   ├── prediction.py               # Node TTF & spatial risk matrix endpoints
│   │   ├── nodes.py                    # Node registry & hardware health diagnostics
│   │   ├── alerts.py                   # Active alerts & SCAMP violation events
│   │   └── simulation.py               # Scenario injection controller
│   ├── core/                           # Geotechnical core logic
│   │   ├── config.py                   # Alert thresholds, grid dimensions, hyperparameters
│   │   ├── conversions.py              # Kinematic calculation (v, a, 1/v, total displacement)
│   │   ├── risk_engine.py              # Multi-parameter rule + ML fusion engine
│   │   ├── spatial.py                  # 3×3 adjacency graph & anomaly cross-validation
│   │   └── health.py                   # Sensor drift, freeze, & heartbeat watchdog
│   ├── services/                       # Stateful business services
│   │   ├── history_service.py          # Per-node isolated 30-sample rolling memory
│   │   ├── prediction_service.py       # LSTM inference pipeline with fallback safety
│   │   └── alert_service.py            # Alert escalation lifecycle management
│   ├── models/schemas.py               # Pydantic validation schemas
│   └── storage/database.py             # SQLite offline ACID storage
│
├── frontend/                           # Mine Safety Operations Dashboard
│   ├── app.py                          # Streamlit real-time monitoring interface
│   └── requirements.txt                # Frontend specific packages
│
├── ml/                                 # AI Modeling & Training Pipeline
│   ├── ttf_lstm_model.keras            # Trained Sequential LSTM neural network
│   ├── lstm_feature_scaler.pkl         # Scaler for kinematic feature vector
│   ├── lstm_target_scaler.pkl          # Target scaler for TTF output (hours)
│   ├── predict_ttf.py                  # Standalone inference wrapper
│   ├── train_lstm.py                   # LSTM training and evaluation script
│   ├── feature_engineering.py          # Feature extraction from creep curves
│   └── all_creep_scenarios.csv         # Synthetic triaxial rock failure dataset
│
├── simulator/                          # Subterranean Mine Telemetry Simulator
│   └── mine_simulator.py               # Real-time multi-scenario generator
│
├── requirements.txt                    # Project dependency specification
├── .gitignore                          # Git exclude rules
└── README.md                           # System documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10 or 3.11 installed.
* PowerShell, Bash, or Command Prompt.

### 2. Installation
Clone the repository and install all required packages:

```bash
git clone https://github.com/ananya01705/STRATA-X-AI-Driven-Real-Time-Mine-Subsidence-Intelligence-Early-Warning-System.git
cd STRATA-X-AI-Driven-Real-Time-Mine-Subsidence-Intelligence-Early-Warning-System
pip install -r requirements.txt
```

---

### 3. Running the System

Open **two terminal windows**:

#### **Terminal 1: Start the Gateway Backend (FastAPI)**
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Swagger API Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
* **API Health Check**: [`http://localhost:8000/health`](http://localhost:8000/health)

#### **Terminal 2: Start the Safety Command Dashboard (Streamlit)**
```powershell
streamlit run frontend/app.py
```
* **Interactive Dashboard**: [`http://localhost:8501`](http://localhost:8501)

---

## 🕹️ Interactive Geotechnical Simulator

Using the sidebar in the Streamlit UI, safety officers and judges can inject live geotechnical scenarios:

| Scenario Code | Description | Expected AI & Safety Response |
| :--- | :--- | :--- |
| `NORMAL` | Background strata creep ($v < 0.05\text{ mm/h}$). | **GREEN / STABLE**: TTF $> 24\text{ h}$. Routine logging. |
| `RISING_DEFORMATION` | Secondary creep acceleration in Gallery 2. | **YELLOW / ADVISORY**: Velocity elevated, maintenance alert logged. |
| `ACCELERATING_DEFORMATION` | Tertiary exponential creep on cluster nodes. | **RED / CRITICAL**: $\text{TTF} < 1.0\text{ h}$, siren evacuation alert triggered. |
| `SENSOR_FAULT` | Hardware disturbance/spike on isolated `NODE_07`. | **FAULT DIAGNOSED**: Spatial engine isolates single node; prevents false alarm. |
| `COMMUNICATION_FAILURE` | Node drop-out & packet loss on `NODE_09`. | **DEGRADED**: Health watchdog flags node for inspection. |

---

## 📡 API Endpoints & Contracts

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/telemetry` | Ingest real-time sensor packet from edge node. |
| `GET` | `/latest` | Fetch latest fused kinematics, TTF prediction, and risk score. |
| `GET` | `/telemetry/history` | Retrieve historical telemetry window for charting. |
| `GET` | `/prediction/{node_id}` | Query dedicated LSTM TTF prediction for a specific node. |
| `GET` | `/risk/map` | Query 3×3 gallery spatial risk matrix with neighbor correlation. |
| `GET` | `/nodes` | List all registered sensor nodes and operational status. |
| `GET` | `/alerts` | Get active SCAMP threshold violation alarms. |
| `POST` | `/simulation/scenario` | Switch geotechnical simulator scenario dynamically. |
| `WS` | `/ws/telemetry` | WebSocket stream for sub-second telemetry broadcast. |

---

## 🛡️ DGMS SCAMP Compliance Matrix

STRATA-X maps real-time multi-sensor telemetry directly to the **Directorate General of Mines Safety (DGMS)** Strata Control and Monitoring Plan (SCAMP) classification:

| Risk Level | Velocity Threshold | Total Deformation | Predicted TTF | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| 🟢 **NORMAL** | $< 0.05\text{ mm/h}$ | $< 5.0\text{ mm}$ | $> 24.0\text{ h}$ | Regular mining operations continue. |
| 🟡 **ADVISORY** | $0.05 - 0.20\text{ mm/h}$ | $5.0 - 15.0\text{ mm}$ | $6.0 - 24.0\text{ h}$ | Increase monitoring frequency; inspect roof bolts. |
| 🟠 **WARNING** | $0.20 - 0.50\text{ mm/h}$ | $15.0 - 25.0\text{ mm}$ | $2.0 - 6.0\text{ h}$ | Halt heavy equipment; reinforce support props. |
| 🔴 **CRITICAL** | $> 0.50\text{ mm/h}$ | $> 25.0\text{ mm}$ | $< 2.0\text{ h}$ | **IMMEDIATE EVACUATION** of miners from gallery. |

---

<div align="center">
Developed with ❤️ for <b>Smart India Hackathon 2026</b>
</div>
