try:
    import pytest  # type: ignore
except ImportError:
    pass

from backend.core.spatial_risk import SpatialRiskEngine, spatial_risk_engine
from backend.models.schemas import RiskState


def test_spatial_topology_and_anchor():
    engine = SpatialRiskEngine()
    assert len(engine.nodes) == 9
    assert engine.nodes["NODE_05"].is_real is True
    assert engine.nodes["NODE_01"].is_real is False
    assert engine.nodes["NODE_05"].grid_x == 1
    assert engine.nodes["NODE_05"].grid_y == 1


def test_neighbor_detection():
    engine = SpatialRiskEngine()
    
    # N5 is in center (1,1); its neighbors in a 3x3 grid should be all 8 surrounding nodes
    n5_neighbors = engine.get_neighbors("NODE_05")
    assert len(n5_neighbors) == 8
    assert "NODE_01" in n5_neighbors
    assert "NODE_09" in n5_neighbors

    # N1 is at top-left (0,0); neighbors are N2 (1,0), N4 (0,1), N5 (1,1)
    n1_neighbors = engine.get_neighbors("NODE_01")
    assert sorted(n1_neighbors) == ["NODE_02", "NODE_04", "NODE_05"]


def test_single_abnormal_node_anomaly_suppression():
    """
    CRITICAL RULE: Single abnormal node must NOT trigger a subsidence zone.
    It must be suppressed as an isolated anomaly.
    """
    engine = SpatialRiskEngine()

    # Only NODE_07 is abnormal (e.g. sensor fault or knock)
    engine.update_node_state(
        "NODE_07",
        deformation_mm=25.0,
        velocity_mm_h=0.35,
        risk_score=65.0,
        risk_level=RiskState.HIGH_RISK
    )

    # All other nodes are NORMAL
    response = engine.analyze_spatial_risk()
    
    assert response.active_zone_count == 0
    assert len(response.zones) == 0
    assert len(response.suppressed_anomalies) == 1
    assert response.suppressed_anomalies[0]["node_id"] == "NODE_07"


def test_connected_multi_node_risk_zone_detection():
    """
    When multiple adjacent nodes exhibit abnormal movement (e.g. N4, N5, N6, N8),
    a subsidence zone MUST be declared.
    """
    engine = SpatialRiskEngine()

    # Set up connected cluster across Gallery 2 + Gallery 3
    engine.update_node_state("NODE_04", deformation_mm=12.0, velocity_mm_h=0.18, predicted_ttf=2.8, risk_score=60.0, risk_level=RiskState.HIGH_RISK)
    engine.update_node_state("NODE_05", deformation_mm=16.0, velocity_mm_h=0.25, predicted_ttf=2.1, risk_score=75.0, risk_level=RiskState.HIGH_RISK)
    engine.update_node_state("NODE_06", deformation_mm=11.5, velocity_mm_h=0.16, predicted_ttf=3.2, risk_score=55.0, risk_level=RiskState.HIGH_RISK)
    engine.update_node_state("NODE_08", deformation_mm=14.0, velocity_mm_h=0.20, predicted_ttf=2.5, risk_score=68.0, risk_level=RiskState.HIGH_RISK)

    response = engine.analyze_spatial_risk()

    assert response.active_zone_count == 1
    assert len(response.zones) == 1
    
    zone = response.zones[0]
    assert zone.zone_id == "ZONE-01"
    assert sorted(zone.affected_node_ids) == ["NODE_04", "NODE_05", "NODE_06", "NODE_08"]
    assert zone.number_of_nodes == 4
    assert zone.risk_level == RiskState.HIGH_RISK
    assert zone.maximum_deformation == 16.0
    assert zone.minimum_ttf == 2.1
    assert zone.contains_real_sensor is True
    assert zone.spatial_confidence >= 0.85
    assert response.mine_risk == RiskState.HIGH_RISK


def test_multiple_independent_risk_clusters():
    """
    If two separate unconnected clusters exist (e.g. Top-Left N1-N2 and Bottom-Right N8-N9),
    two separate zones must be identified.
    """
    engine = SpatialRiskEngine()

    # Cluster 1: N1 (0,0) and N2 (1,0) (Top row)
    engine.update_node_state("NODE_01", deformation_mm=10.0, velocity_mm_h=0.15, risk_score=58.0, risk_level=RiskState.HIGH_RISK)
    engine.update_node_state("NODE_02", deformation_mm=11.0, velocity_mm_h=0.16, risk_score=60.0, risk_level=RiskState.HIGH_RISK)

    # Cluster 2: N8 (1,2) and N9 (2,2) (Bottom row)
    engine.update_node_state("NODE_08", deformation_mm=9.5, velocity_mm_h=0.14, risk_score=56.0, risk_level=RiskState.HIGH_RISK)
    engine.update_node_state("NODE_09", deformation_mm=10.5, velocity_mm_h=0.15, risk_score=59.0, risk_level=RiskState.HIGH_RISK)

    response = engine.analyze_spatial_risk()

    assert response.active_zone_count == 2
    assert len(response.zones) == 2
    zone_ids = [z.zone_id for z in response.zones]
    assert "ZONE-01" in zone_ids
    assert "ZONE-02" in zone_ids
