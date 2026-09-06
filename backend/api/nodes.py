from fastapi import APIRouter, HTTPException, Path
from typing import List, Dict, Any
from datetime import datetime

from backend.storage.database import db
from backend.services.history_service import history_service
from backend.services.prediction_service import prediction_service
from backend.core.risk_engine import risk_engine
from backend.core.health import health_engine
from backend.core.config import DEFAULT_MINE_NODES
from backend.models.schemas import NodeInfo, RiskState

router = APIRouter(prefix="/nodes", tags=["Nodes & Health"])


@router.get("", response_model=List[NodeInfo])
def get_all_nodes():
    """Returns list of all sensor nodes with live status, battery, buffer count, and risk."""
    db_nodes = {n["node_id"]: n for n in db.get_all_nodes()}
    latest_readings = history_service.get_all_latest_readings()
    result = []

    for nid, default_info in DEFAULT_MINE_NODES.items():
        db_n = db_nodes.get(nid, {})
        reading = latest_readings.get(nid, {})
        
        pred = prediction_service.predict_node_ttf(nid)
        risk = risk_engine.assess_node_risk(nid, pred, latest_readings)
        health_status = health_engine.evaluate_node_health(reading)

        result.append(NodeInfo(
            node_id=nid,
            name=default_info["name"],
            grid_x=default_info["grid_x"],
            grid_y=default_info["grid_y"],
            latitude=default_info["lat"],
            longitude=default_info["lon"],
            depth_m=default_info["depth_m"],
            status=health_status,
            battery=float(reading.get("battery", db_n.get("battery", 100.0))),
            signal_strength=float(reading.get("signal_strength", db_n.get("signal_strength", -70.0))),
            last_seen=reading.get("timestamp"),
            readings_buffered=history_service.get_node_buffer_count(nid),
            current_risk=risk.risk_state,
            current_ttf_hours=pred.get("ttf_hours")
        ))

    return result


@router.get("/{node_id}")
def get_node_details(node_id: str = Path(..., description="Sensor Node ID")):
    """Returns comprehensive details, live metrics, and recent history for one node."""
    if node_id not in DEFAULT_MINE_NODES:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found.")

    node_def = DEFAULT_MINE_NODES[node_id]
    latest = history_service.get_latest_reading(node_id) or {}
    pred = prediction_service.predict_node_ttf(node_id)
    risk = risk_engine.assess_node_risk(node_id, pred)
    history = db.get_node_telemetry_history(node_id, limit=30)

    return {
        "node_id": node_id,
        "metadata": node_def,
        "live_telemetry": latest,
        "prediction": pred,
        "risk_assessment": risk.dict(),
        "recent_readings": history
    }


@router.get("/{node_id}/health")
def get_node_health(node_id: str = Path(..., description="Sensor Node ID")):
    """Returns diagnostics on battery, signal, communication latency, and fault flags."""
    latest = history_service.get_latest_reading(node_id) or {}
    status = health_engine.evaluate_node_health(latest)
    all_readings = history_service.get_all_latest_readings()
    anomaly = health_engine.detect_sensor_fault_vs_movement(node_id, all_readings)

    return {
        "node_id": node_id,
        "health_status": status,
        "battery_pct": latest.get("battery", 100.0),
        "signal_strength_dbm": latest.get("signal_strength", -70.0),
        "buffer_occupancy": f"{history_service.get_node_buffer_count(node_id)}/30",
        "anomaly_diagnostics": anomaly.dict()
    }
