from fastapi import APIRouter, Path
from datetime import datetime
from typing import Dict, Any, List

from backend.services.prediction_service import prediction_service
from backend.services.history_service import history_service
from backend.core.risk_engine import risk_engine
from backend.core.spatial_risk import spatial_risk_engine
from backend.core.config import DEFAULT_MINE_NODES, REAL_SENSOR_NODE_ID
from backend.core.exceptions import NodeNotFoundException
from backend.models.schemas import RiskState

router = APIRouter(tags=["Prediction & Risk"])


@router.get("/prediction/{node_id}")
def get_node_prediction(node_id: str = Path(..., description="Sensor Node ID")):
    """Returns Time-to-Failure (TTF) prediction and kinematics for a specific node."""
    if node_id not in DEFAULT_MINE_NODES:
        raise NodeNotFoundException(node_id)
    return prediction_service.predict_node_ttf(node_id)


@router.get("/risk")
def get_overall_mine_risk():
    """
    Returns the comprehensive mine-wide risk assessment,
    aggregating individual node risks and Layer 2 spatial correlation.
    """
    latest_readings = history_service.get_all_latest_readings()
    
    node_assessments: List[Dict[str, Any]] = []
    max_risk_score = 0.0
    critical_nodes = []
    min_ttf = None

    for nid in DEFAULT_MINE_NODES.keys():
        pred = prediction_service.predict_node_ttf(nid)
        assessment = risk_engine.assess_node_risk(nid, pred, latest_readings)
        
        node_assessments.append({
            "node_id": nid,
            "is_real": (nid == REAL_SENSOR_NODE_ID),
            "risk_score": assessment.risk_score,
            "risk_state": assessment.risk_state.value,
            "ttf_hours": assessment.ttf_hours,
            "trend": assessment.trend.value,
            "confidence": assessment.confidence,
            "anomaly": assessment.anomaly.dict() if assessment.anomaly else None
        })

        if assessment.risk_score > max_risk_score:
            max_risk_score = assessment.risk_score

        if assessment.risk_state == RiskState.CRITICAL:
            critical_nodes.append(nid)

        if assessment.ttf_hours is not None:
            if min_ttf is None or assessment.ttf_hours < min_ttf:
                min_ttf = assessment.ttf_hours

    # Layer 2 spatial risk synthesis
    spatial_map = spatial_risk_engine.analyze_spatial_risk(latest_readings=latest_readings)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_risk_score": max_risk_score,
        "overall_risk_state": spatial_map.mine_risk.value,
        "lowest_ttf_hours": min_ttf,
        "critical_nodes_count": len(critical_nodes),
        "critical_nodes": critical_nodes,
        "active_zones_count": spatial_map.active_zone_count,
        "node_assessments": node_assessments
    }


@router.get("/risk/map")
def get_spatial_risk_map():
    """Returns spatial grid heatmap coordinates and risk data for map visualization."""
    latest_readings = history_service.get_all_latest_readings()
    grid_points = []

    for nid, info in DEFAULT_MINE_NODES.items():
        pred = prediction_service.predict_node_ttf(nid)
        risk = risk_engine.assess_node_risk(nid, pred, latest_readings)
        reading = latest_readings.get(nid, {})

        grid_points.append({
            "node_id": nid,
            "name": info["name"],
            "grid_x": info["grid_x"],
            "grid_y": info["grid_y"],
            "lat": info["lat"],
            "lon": info["lon"],
            "depth_m": info["depth_m"],
            "is_real": info.get("is_real", nid == REAL_SENSOR_NODE_ID),
            "tilt_deg": reading.get("tilt_deg", 1.0),
            "deformation_mm": pred.get("deformation_mm", 0.0),
            "velocity_mm_h": pred.get("velocity_mm_h", 0.04),
            "ttf_hours": pred.get("ttf_hours"),
            "risk_score": risk.risk_score,
            "risk_state": risk.risk_state.value,
            "is_anomaly": risk.anomaly.is_anomaly if risk.anomaly else False
        })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_nodes": len(grid_points),
        "nodes": grid_points
    }
