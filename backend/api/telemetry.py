from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from backend.models.schemas import SensorReading, TelemetryResponse
from backend.services.history_service import history_service
from backend.services.prediction_service import prediction_service
from backend.core.risk_engine import risk_engine
from backend.services.alert_service import alert_service
from backend.storage.database import db

router = APIRouter(tags=["Telemetry"])


@router.post("/telemetry", response_model=TelemetryResponse)
def receive_telemetry(reading: SensorReading):
    """
    Ingestion endpoint for raw LoRa / ESP32 sensor telemetry.
    Stores telemetry in SQLite database, updates the node's 30-reading rolling window,
    and runs TTF prediction + risk assessment.
    """
    data = reading.model_dump() if hasattr(reading, "model_dump") else reading.dict()

    
    # 1. Record raw reading in local SQLite DB
    db.record_telemetry(data)

    # 2. Append to rolling window buffer
    buffer_info = history_service.append_reading(data)

    # 3. If buffer has accumulated sufficient data, perform TTF inference & risk assessment
    if buffer_info["readings_count"] >= 30:
        pred = prediction_service.predict_node_ttf(reading.node_id)
        if pred.get("status") == "READY":
            risk = risk_engine.assess_node_risk(reading.node_id, pred)
            db.record_prediction(pred, risk.dict())
            alert_service.process_node_assessment(risk)

    return TelemetryResponse(
        accepted=True,
        node_id=reading.node_id,
        timestamp=reading.timestamp,
        buffer_status=buffer_info["buffer_status"],
        readings_count=buffer_info["readings_count"],
        required_readings=buffer_info["required_readings"]
    )


@router.get("/latest")
def get_latest_telemetry():
    """
    Returns the most recent system telemetry and risk state.
    Fully compatible with Member 3's frontend dashboard contract.
    """
    latest = history_service.get_latest_reading()
    
    if not latest:
        # Initial cold start default
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": "NODE_01",
            "tilt_deg": 1.25,
            "ttf_hours": 8.50,
            "velocity_mm_h": 0.045,
            "deformation_mm": 6.20,
            "risk_status": "NORMAL",
            "risk_score": 14.5,
            "confidence": 0.95,
            "buffer_status": "INITIALIZING"
        }

    node_id = latest["node_id"]
    pred = prediction_service.predict_node_ttf(node_id)
    risk = risk_engine.assess_node_risk(node_id, pred)

    return {
        "timestamp": latest.get("timestamp", datetime.utcnow().isoformat()),
        "node_id": node_id,
        "tilt_deg": round(float(latest.get("tilt_deg", 1.0)), 2),
        "ttf_hours": pred.get("ttf_hours", 8.5),
        "velocity_mm_h": pred.get("velocity_mm_h", 0.04),
        "deformation_mm": pred.get("deformation_mm", 5.0),
        "risk_status": risk.risk_state.value,
        "risk_score": risk.risk_score,
        "confidence": risk.confidence,
        "trend": risk.trend.value,
        "buffer_status": pred.get("buffer_progress", "READY")
    }


@router.get("/telemetry/history")
def get_telemetry_history(
    node_id: str = Query("NODE_01", description="Node identifier"),
    limit: int = Query(50, description="Max historical points")
):
    """Returns historical raw telemetry points for chart plotting."""
    return db.get_node_telemetry_history(node_id=node_id, limit=limit)
