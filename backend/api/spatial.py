from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime

from backend.core.spatial_risk import spatial_risk_engine
from backend.services.history_service import history_service
from backend.services.prediction_service import prediction_service
from backend.core.risk_engine import risk_engine
from backend.core.config import DEFAULT_MINE_NODES
from backend.models.schemas import (
    SpatialRiskZone,
    SpatialRiskMapResponse,
    SpatialNodesResponse,
)

router = APIRouter(prefix="/spatial", tags=["Layer 2: Spatial Risk Intelligence"])


def _compute_current_spatial_state() -> SpatialRiskMapResponse:
    """Helper to synthesize live Layer 1 predictions into Layer 2 spatial risk map."""
    latest_readings = history_service.get_all_latest_readings()
    node_assessments: Dict[str, Any] = {}

    for nid in DEFAULT_MINE_NODES.keys():
        pred = prediction_service.predict_node_ttf(nid)
        risk = risk_engine.assess_node_risk(nid, pred, latest_readings)
        node_assessments[nid] = {
            "risk_score": risk.risk_score,
            "risk_state": risk.risk_state.value,
            "ttf_hours": risk.ttf_hours,
            "trend": risk.trend.value,
            "confidence": risk.confidence,
            "anomaly": risk.anomaly.dict() if risk.anomaly else None,
        }

    return spatial_risk_engine.analyze_spatial_risk(
        node_assessments=node_assessments,
        latest_readings=latest_readings,
    )


@router.get("/nodes", response_model=SpatialNodesResponse)
def get_spatial_nodes():
    """
    Returns spatial coordinate topology and live geotechnical state for all 9 sensor nodes.
    Explicitly distinguishes physical hardware (N5) from software simulated prototype nodes.
    """
    spatial_map = _compute_current_spatial_state()
    nodes = spatial_map.nodes

    real_count = sum(1 for n in nodes if n.get("is_real"))
    sim_count = len(nodes) - real_count

    return SpatialNodesResponse(
        timestamp=datetime.utcnow(),
        total_nodes=len(nodes),
        real_nodes_count=real_count,
        simulated_nodes_count=sim_count,
        nodes=nodes,
    )


@router.get("/zones", response_model=List[SpatialRiskZone])
def get_active_spatial_risk_zones():
    """
    Returns all actively detected multi-node subsidence hazard zones.
    Isolated sensor anomalies are suppressed and will not appear as zones.
    """
    spatial_map = _compute_current_spatial_state()
    return spatial_map.zones


@router.get("/risk-map", response_model=SpatialRiskMapResponse)
def get_full_spatial_risk_map():
    """
    Comprehensive Layer 2 Spatial Risk Map response:
    Combines 3x3 node coordinates, active connected risk zones, isolated anomaly suppressions,
    and overall mine-level risk synthesis.
    """
    return _compute_current_spatial_state()
