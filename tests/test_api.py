try:
    import pytest  # type: ignore
except ImportError:
    pass

from datetime import datetime


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "STRATA-X — AI-Driven Mine Subsidence Intelligence"
    assert data["real_sensor_anchor"] == "NODE_05"


def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "spatial_engine" in data


def test_get_system_status(client):
    response = client.get("/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["system_online"] is True
    assert data["total_nodes"] == 9
    assert data["real_sensor_node"] == "NODE_05"


def test_post_telemetry_valid(client):
    payload = {
        "node_id": "NODE_05",
        "tilt_x": 0.8,
        "tilt_y": 0.6,
        "acceleration": 0.02,
        "displacement_mm": 2.5,
        "vibration": 0.03,
        "battery": 95.0,
        "signal_strength": -68.0,
        "is_real": True
    }
    response = client.post("/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is True
    assert data["node_id"] == "NODE_05"
    assert data["is_real"] is True
    assert data["readings_count"] == 1


def test_post_telemetry_invalid(client):
    # Missing required node_id
    payload = {"tilt_x": 1.0}
    response = client.post("/telemetry", json=payload)
    assert response.status_code == 422


def test_get_latest(client):
    response = client.get("/latest")
    assert response.status_code == 200
    data = response.json()
    assert "node_id" in data
    assert "risk_status" in data


def test_get_nodes(client):
    response = client.get("/nodes")
    assert response.status_code == 200
    nodes = response.json()
    assert len(nodes) == 9
    n5 = next(n for n in nodes if n["node_id"] == "NODE_05")
    assert n5["is_real"] is True


def test_get_node_details_valid(client):
    response = client.get("/nodes/NODE_05")
    assert response.status_code == 200
    data = response.json()
    assert data["node_id"] == "NODE_05"
    assert data["is_real"] is True


def test_get_node_details_invalid_404(client):
    response = client.get("/nodes/NODE_INVALID_99")
    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "ERR_NODE_NOT_FOUND"
    assert "error" in data


def test_get_spatial_endpoints(client):
    # /spatial/nodes
    resp_nodes = client.get("/spatial/nodes")
    assert resp_nodes.status_code == 200
    data_nodes = resp_nodes.json()
    assert data_nodes["total_nodes"] == 9
    assert data_nodes["real_nodes_count"] == 1
    assert data_nodes["simulated_nodes_count"] == 8

    # /spatial/zones
    resp_zones = client.get("/spatial/zones")
    assert resp_zones.status_code == 200
    assert isinstance(resp_zones.json(), list)

    # /spatial/risk-map
    resp_map = client.get("/spatial/risk-map")
    assert resp_map.status_code == 200
    data_map = resp_map.json()
    assert "mine_risk" in data_map
    assert "active_zone_count" in data_map
    assert "zones" in data_map
    assert "nodes" in data_map
