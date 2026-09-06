"""
STRATA-X — Layer 2: Deterministic Spatial Risk Engine
=====================================================
Evaluates underground mine sensor topology across the 3x3 gallery grid.
Distinguishes isolated sensor faults from genuine, multi-node spatially correlated subsidence zones.

Topology Grid (3x3):
--------------------
N1 (0,0)  |  N2 (1,0)  |  N3 (2,0)
N4 (0,1)  |  N5 (1,1)* |  N6 (2,1)
N7 (0,2)  |  N8 (1,2)  |  N9 (2,2)

* N5 (Center Junction) = Real Physical Sensor Hardware (is_real = True)
* N1-N4, N6-N9 = Simulated Prototype Nodes (is_real = False)
"""

import math
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime

from backend.core.config import (
    DEFAULT_MINE_NODES,
    TTF_CRITICAL_HOURS,
    TTF_HIGH_RISK_HOURS,
    TTF_WATCH_HOURS,
    VELOCITY_THRESHOLD_HIGH,
)
from backend.models.schemas import (
    RiskState,
    SpatialRiskZone,
    SpatialRiskMapResponse,
)

logger = logging.getLogger("SpatialRiskEngine")


class SpatialNode:
    """Represents a single node in the spatial grid with its latest geotechnical state."""

    def __init__(
        self,
        node_id: str,
        grid_x: int,
        grid_y: int,
        lat: float,
        lon: float,
        depth_m: float,
        is_real: bool = False,
        name: str = "",
    ):
        self.node_id = node_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.lat = lat
        self.lon = lon
        self.depth_m = depth_m
        self.is_real = is_real
        self.name = name or node_id

        # Live state
        self.tilt_deg: float = 0.0
        self.deformation_mm: float = 0.0
        self.velocity_mm_h: float = 0.0
        self.inverse_velocity: float = 0.0
        self.predicted_ttf: Optional[float] = None
        self.risk_score: float = 0.0
        self.risk_level: RiskState = RiskState.NORMAL
        self.is_anomaly: bool = False
        self.anomaly_type: Optional[str] = None
        self.last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "lat": self.lat,
            "lon": self.lon,
            "depth_m": self.depth_m,
            "is_real": self.is_real,
            "tilt_deg": round(self.tilt_deg, 2),
            "deformation_mm": round(self.deformation_mm, 2),
            "velocity_mm_h": round(self.velocity_mm_h, 4),
            "inverse_velocity": round(self.inverse_velocity, 4),
            "predicted_ttf": round(self.predicted_ttf, 2) if self.predicted_ttf is not None else None,
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level.value,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class SpatialRiskEngine:
    """
    Layer 2 Deterministic Spatial Risk Engine.

    Responsibilities:
    1. Node coordinate & neighborhood topology modeling.
    2. Detection of abnormal nodes based on Layer 1 risk, TTF, and kinematic thresholds.
    3. Isolated sensor anomaly suppression:
       - Single abnormal node + normal neighbors -> Suppressed (NO SUBSIDENCE ZONE).
    4. Connected risk cluster detection:
       - 2+ adjacent abnormal nodes -> Form active subsidence risk zone (e.g. ZONE-01).
    5. Zone statistics computation (centroid, average/max deformation, min TTF, confidence).
    6. Transparent tagging of real hardware (N5) vs simulation (N1-N4, N6-N9).
    """

    def __init__(self, node_definitions: Dict[str, Dict[str, Any]] = DEFAULT_MINE_NODES):
        self.node_definitions = node_definitions
        self.nodes: Dict[str, SpatialNode] = {}
        self._init_topology()

    def _init_topology(self):
        """Initializes spatial nodes with coordinates and hardware/simulation roles."""
        self.nodes = {}
        for nid, meta in self.node_definitions.items():
            # N5 is the physical anchor; all other nodes are software simulations
            is_real = meta.get("is_real", nid == "NODE_05")
            node = SpatialNode(
                node_id=nid,
                grid_x=int(meta.get("grid_x", 0)),
                grid_y=int(meta.get("grid_y", 0)),
                lat=float(meta.get("lat", 23.6345)),
                lon=float(meta.get("lon", 85.2799)),
                depth_m=float(meta.get("depth_m", 120.0)),
                is_real=is_real,
                name=meta.get("name", nid),
            )
            self.nodes[nid] = node

    def get_spatial_distance(self, node_a_id: str, node_b_id: str) -> float:
        """Calculates Euclidean distance between two nodes in grid units."""
        if node_a_id not in self.nodes or node_b_id not in self.nodes:
            return float("inf")
        na, nb = self.nodes[node_a_id], self.nodes[node_b_id]
        return math.sqrt((na.grid_x - nb.grid_x) ** 2 + (na.grid_y - nb.grid_y) ** 2)

    def get_neighbors(self, node_id: str, max_distance: float = 1.5) -> List[str]:
        """
        Returns adjacent neighbor node IDs.
        For a 3x3 grid, max_distance=1.5 includes both orthogonal (dist=1.0)
        and diagonal (dist=1.414) adjacent gallery nodes.
        """
        if node_id not in self.nodes:
            return []
        neighbors = []
        for other_id in self.nodes.keys():
            if other_id == node_id:
                continue
            if self.get_spatial_distance(node_id, other_id) <= max_distance:
                neighbors.append(other_id)
        return sorted(neighbors)

    def update_node_state(
        self,
        node_id: str,
        tilt_deg: float = 0.0,
        deformation_mm: float = 0.0,
        velocity_mm_h: float = 0.0,
        predicted_ttf: Optional[float] = None,
        risk_score: float = 0.0,
        risk_level: RiskState = RiskState.NORMAL,
        is_anomaly: bool = False,
        anomaly_type: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Updates live geotechnical metrics for a specific node."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.tilt_deg = tilt_deg
        node.deformation_mm = deformation_mm
        node.velocity_mm_h = velocity_mm_h
        node.inverse_velocity = 1.0 / max(velocity_mm_h, 1e-4)
        node.predicted_ttf = predicted_ttf
        node.risk_score = risk_score
        node.risk_level = risk_level
        node.is_anomaly = is_anomaly
        node.anomaly_type = anomaly_type
        node.last_updated = timestamp or datetime.utcnow()

    def is_node_abnormal(self, node: SpatialNode) -> bool:
        """
        Determines if a node exhibits abnormal deformation or elevated risk.
        A node is considered abnormal if:
        - Risk state is WATCH, HIGH_RISK, or CRITICAL
        - Risk score >= 35.0
        - TTF <= 6.0 hours (if TTF is ready)
        - Velocity >= 0.10 mm/h
        - Deformation >= 8.0 mm
        """
        if node.risk_level in (RiskState.WATCH, RiskState.HIGH_RISK, RiskState.CRITICAL):
            return True
        if node.risk_score >= 35.0:
            return True
        if node.predicted_ttf is not None and node.predicted_ttf <= TTF_WATCH_HOURS:
            return True
        if node.velocity_mm_h >= 0.10:
            return True
        if node.deformation_mm >= 8.0:
            return True
        return False

    def detect_connected_clusters(self, abnormal_node_ids: Set[str]) -> List[List[str]]:
        """
        Identifies connected clusters of abnormal nodes using Breadth-First Search (BFS).
        Two abnormal nodes are connected if they are adjacent (grid distance <= 1.5).
        """
        visited: Set[str] = set()
        clusters: List[List[str]] = []

        for nid in sorted(abnormal_node_ids):
            if nid in visited:
                continue

            # Start BFS from this unvisited abnormal node
            cluster: List[str] = []
            queue = [nid]
            visited.add(nid)

            while queue:
                current = queue.pop(0)
                cluster.append(current)

                # Check all spatial neighbors
                neighbors = self.get_neighbors(current, max_distance=1.5)
                for neighbor_id in neighbors:
                    if neighbor_id in abnormal_node_ids and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            clusters.append(sorted(cluster))

        return clusters

    def analyze_spatial_risk(
        self,
        node_assessments: Optional[Dict[str, Any]] = None,
        latest_readings: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SpatialRiskMapResponse:
        """
        Main Layer 2 Analysis Pipeline:
        1. Ingests latest node states (from history & risk engine).
        2. Identifies abnormal nodes.
        3. Applies isolated anomaly suppression.
        4. Detects multi-node connected clusters and generates SpatialRiskZone objects.
        5. Computes comprehensive mine-level risk synthesis.
        """
        # 1. Update internal node states if latest assessments/readings are provided
        if latest_readings:
            for nid, data in latest_readings.items():
                if nid in self.nodes:
                    self.nodes[nid].tilt_deg = float(data.get("tilt_deg", self.nodes[nid].tilt_deg))
                    self.nodes[nid].deformation_mm = float(data.get("deformation", data.get("displacement_mm", self.nodes[nid].deformation_mm)))
                    self.nodes[nid].velocity_mm_h = float(data.get("velocity", self.nodes[nid].velocity_mm_h))

        if node_assessments:
            for nid, assess in node_assessments.items():
                if nid in self.nodes:
                    self.nodes[nid].risk_score = float(assess.get("risk_score", self.nodes[nid].risk_score))
                    r_level_str = assess.get("risk_state", "NORMAL")
                    try:
                        self.nodes[nid].risk_level = RiskState(r_level_str)
                    except ValueError:
                        self.nodes[nid].risk_level = RiskState.NORMAL
                    self.nodes[nid].predicted_ttf = assess.get("ttf_hours")
                    anomaly_data = assess.get("anomaly")
                    if anomaly_data:
                        self.nodes[nid].is_anomaly = anomaly_data.get("is_anomaly", False)
                        self.nodes[nid].anomaly_type = anomaly_data.get("anomaly_type")

        # 2. Find all currently abnormal nodes
        abnormal_nodes: Set[str] = set()
        for nid, node in self.nodes.items():
            if self.is_node_abnormal(node):
                abnormal_nodes.add(nid)

        # 3. Detect connected clusters of abnormal nodes
        raw_clusters = self.detect_connected_clusters(abnormal_nodes)

        # 4. Filter & Classify Clusters into Risk Zones vs Isolated Anomalies
        active_zones: List[SpatialRiskZone] = []
        suppressed_anomalies: List[Dict[str, Any]] = []

        zone_idx = 1
        for cluster in raw_clusters:
            # Rule: Single abnormal node is treated as an isolated anomaly, NOT a subsidence zone
            if len(cluster) == 1:
                single_node_id = cluster[0]
                single_node = self.nodes[single_node_id]
                neighbors = self.get_neighbors(single_node_id)
                normal_neighbors = [n for n in neighbors if n not in abnormal_nodes]

                # Suppress zone creation
                suppressed_anomalies.append({
                    "node_id": single_node_id,
                    "reason": "Single abnormal node with normal surrounding gallery nodes. Suppressed zone creation.",
                    "neighbors_normal": normal_neighbors,
                    "confidence": 0.85,
                })
                logger.info(
                    f"Layer 2 Anomaly Suppression: Isolated node {single_node_id} "
                    f"(risk: {single_node.risk_level.value}) suppressed. No multi-node subsidence zone declared."
                )
                continue

            # If 2 or more connected nodes are abnormal, this is a genuine subsidence zone
            zone_id = f"ZONE-{zone_idx:02d}"
            zone_idx += 1

            cluster_nodes = [self.nodes[nid] for nid in cluster]

            # Compute Centroid
            avg_x = sum(n.grid_x for n in cluster_nodes) / len(cluster_nodes)
            avg_y = sum(n.grid_y for n in cluster_nodes) / len(cluster_nodes)
            avg_lat = sum(n.lat for n in cluster_nodes) / len(cluster_nodes)
            avg_lon = sum(n.lon for n in cluster_nodes) / len(cluster_nodes)

            # Compute Deformations
            deformations = [n.deformation_mm for n in cluster_nodes]
            avg_def = sum(deformations) / len(deformations)
            max_def = max(deformations)

            # Compute Velocities
            velocities = [n.velocity_mm_h for n in cluster_nodes]
            avg_vel = sum(velocities) / len(velocities)
            max_vel = max(velocities)

            # Compute TTF
            valid_ttfs = [n.predicted_ttf for n in cluster_nodes if n.predicted_ttf is not None]
            avg_ttf = (sum(valid_ttfs) / len(valid_ttfs)) if valid_ttfs else None
            min_ttf = min(valid_ttfs) if valid_ttfs else None

            # Determine Zone Risk Level
            max_risk_score = max(n.risk_score for n in cluster_nodes)
            has_critical = any(n.risk_level == RiskState.CRITICAL for n in cluster_nodes)
            has_high = any(n.risk_level == RiskState.HIGH_RISK for n in cluster_nodes)

            if has_critical or (min_ttf is not None and min_ttf <= TTF_CRITICAL_HOURS) or max_risk_score >= 80.0:
                zone_risk = RiskState.CRITICAL
            elif has_high or (min_ttf is not None and min_ttf <= TTF_HIGH_RISK_HOURS) or max_risk_score >= 55.0:
                zone_risk = RiskState.HIGH_RISK
            else:
                zone_risk = RiskState.WATCH

            # Spatial Confidence: Based on cluster size, continuity, and real anchor presence
            base_conf = 0.80 + min(len(cluster) * 0.04, 0.15)
            if any(n.is_real for n in cluster_nodes):
                base_conf += 0.04  # Extra confidence when real physical sensor N5 is included
            spatial_confidence = round(min(base_conf, 0.98), 2)

            # Explainable Evidence Reasons
            reasons = [
                f"Spatially correlated movement verified across {len(cluster)} adjacent nodes ({', '.join(cluster)}).",
                f"Maximum strata deformation: {max_def:.2f} mm (cluster avg: {avg_def:.2f} mm).",
                f"Peak deformation velocity: {max_vel:.3f} mm/h.",
            ]
            if min_ttf is not None:
                reasons.append(f"Lowest predicted Time-to-Failure: {min_ttf:.2f} hours.")
            if any(n.is_real for n in cluster_nodes):
                reasons.append("Physical anchor node N5 actively participating in deformation zone.")

            active_zones.append(SpatialRiskZone(
                zone_id=zone_id,
                affected_node_ids=cluster,
                number_of_nodes=len(cluster),
                centroid={"grid_x": round(avg_x, 2), "grid_y": round(avg_y, 2), "lat": avg_lat, "lon": avg_lon},
                average_deformation=round(avg_def, 2),
                maximum_deformation=round(max_def, 2),
                average_velocity=round(avg_vel, 4),
                maximum_velocity=round(max_vel, 4),
                average_ttf=round(avg_ttf, 2) if avg_ttf is not None else None,
                minimum_ttf=round(min_ttf, 2) if min_ttf is not None else None,
                risk_level=zone_risk,
                spatial_confidence=spatial_confidence,
                reasons=reasons,
                contains_real_sensor=any(n.is_real for n in cluster_nodes),
            ))

        # 5. Determine Overall Mine-Level Risk State
        if any(z.risk_level == RiskState.CRITICAL for z in active_zones):
            mine_risk = RiskState.CRITICAL
        elif any(z.risk_level == RiskState.HIGH_RISK for z in active_zones):
            mine_risk = RiskState.HIGH_RISK
        elif any(z.risk_level == RiskState.WATCH for z in active_zones):
            mine_risk = RiskState.WATCH
        elif len(abnormal_nodes) > 0:
            # Isolated anomaly exists without zone
            mine_risk = RiskState.WATCH
        else:
            mine_risk = RiskState.NORMAL

        all_nodes_list = [n.to_dict() for n in self.nodes.values()]

        return SpatialRiskMapResponse(
            timestamp=datetime.utcnow(),
            mine_risk=mine_risk,
            active_zone_count=len(active_zones),
            zones=active_zones,
            nodes=all_nodes_list,
            suppressed_anomalies=suppressed_anomalies,
            total_nodes=len(self.nodes),
            real_sensor_id="NODE_05",
        )


# Global Layer 2 Singleton
spatial_risk_engine = SpatialRiskEngine()
