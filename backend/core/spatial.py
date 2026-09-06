import math
from typing import Dict, List, Tuple, Any
from backend.core.config import DEFAULT_MINE_NODES


class SpatialGraph:
    """
    Represents the spatial layout and neighborhood topology of underground mine sensor nodes.
    Used for spatial correlation analysis and isolated anomaly detection.
    """

    def __init__(self, node_definitions: Dict[str, Dict[str, Any]] = DEFAULT_MINE_NODES):
        self.nodes = node_definitions

    def get_neighbors(self, node_id: str, max_distance: float = 1.5) -> List[str]:
        """
        Returns adjacent neighbor node IDs within grid distance threshold.
        For a 3x3 grid, Manhattan / Chebyshev distance <= 1.5 gives immediate neighbors.
        """
        if node_id not in self.nodes:
            return []

        target_node = self.nodes[node_id]
        tx, ty = target_node["grid_x"], target_node["grid_y"]
        neighbors = []

        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            ox, oy = other_node["grid_x"], other_node["grid_y"]
            dist = math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2)
            if dist <= max_distance:
                neighbors.append(other_id)

        return neighbors

    def calculate_spatial_correlation(
        self,
        node_id: str,
        node_velocities: Dict[str, float]
    ) -> Tuple[float, List[str], str]:
        """
        Calculates how correlated this node's movement is with its spatial neighbors.
        
        Returns:
            (correlation_score, active_neighbors, interpretation)
            correlation_score: 0.0 (totally isolated) to 1.0 (strongly correlated across gallery)
        """
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.5, [], "No spatial neighbors registered"

        target_vel = node_velocities.get(node_id, 0.0)
        neighbor_vels = [node_velocities.get(nid, 0.0) for nid in neighbors if nid in node_velocities]

        if not neighbor_vels:
            return 0.5, [], "No active neighbor readings"

        avg_neighbor_vel = sum(neighbor_vels) / len(neighbor_vels)

        # Ratio of target velocity vs neighbor average
        if target_vel > 0.15 and avg_neighbor_vel < 0.05:
            # Target is moving rapidly, but neighbors are completely quiet -> Isolated
            correlation = 0.15
            interpretation = "ISOLATED_ANOMALY: High movement on node while neighboring gallery is stationary."
        elif target_vel > 0.10 and avg_neighbor_vel > 0.08:
            # Both target and neighbors are moving -> Coherent strata movement
            correlation = 0.92
            interpretation = "CORRELATED_STRATA_MOVEMENT: Coordinated displacement observed across adjacent galleries."
        else:
            correlation = 0.70
            interpretation = "HOMOGENEOUS_BASELINE: Normal uniform strata background readings."

        return correlation, neighbors, interpretation


# Global Singleton
spatial_graph = SpatialGraph()
