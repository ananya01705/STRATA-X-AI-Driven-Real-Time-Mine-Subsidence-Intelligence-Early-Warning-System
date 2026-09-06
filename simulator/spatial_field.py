"""
STRATA-X — Spatial Field Telemetry Simulator
=============================================
Simulates spatially correlated geotechnical telemetry across the 3x3 mine gallery mesh.

Physical / Simulated Role:
- N5 (Gallery 2 - Center Junction) = REAL PHYSICAL SENSOR ANCHOR (is_real = True)
- N1-N4, N6-N9 = SOFTWARE SIMULATED PROTOTYPE NODES (is_real = False)

Spatial Propagation Principle:
Deformation and strain energy propagate outward from the anchor/epicenter (N5)
based on Euclidean distance with exponential decay:
  response_factor = exp(-decay_rate * distance)
Nearby gallery nodes (N2, N4, N6, N8 at dist=1.0) respond with high-to-moderate magnitude.
Corner nodes (N1, N3, N7, N9 at dist=1.414) experience attenuated sympathetic displacement.
"""

import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.core.config import DEFAULT_MINE_NODES
from backend.models.schemas import SimulationScenario


class SpatialFieldSimulator:
    """
    Physics-informed multi-node spatial field telemetry generator.
    Ensures simulated prototype nodes exhibit authentic geotechnical spatial correlation
    with the physical anchor sensor N5, rather than independent random noise.
    """

    def __init__(self):
        self.scenario = SimulationScenario.NORMAL
        self.step_count = 0
        self.anchor_node_id = "NODE_05"
        self.node_states: Dict[str, Dict[str, Any]] = {}
        self.init_field()

    def init_field(self):
        """Initializes all 9 gallery nodes to stable baseline state."""
        self.step_count = 0
        self.node_states = {}
        for nid, meta in DEFAULT_MINE_NODES.items():
            is_real = meta.get("is_real", nid == self.anchor_node_id)
            gx, gy = meta.get("grid_x", 0), meta.get("grid_y", 0)
            
            # Distance from physical anchor N5 (1,1)
            dist_to_anchor = math.sqrt((gx - 1) ** 2 + (gy - 1) ** 2)

            self.node_states[nid] = {
                "node_id": nid,
                "is_real": is_real,
                "grid_x": gx,
                "grid_y": gy,
                "dist_to_anchor": dist_to_anchor,
                "base_tilt_x": round(random.uniform(0.6, 1.1), 2),
                "base_tilt_y": round(random.uniform(0.4, 0.9), 2),
                "displacement_mm": round(random.uniform(1.2, 2.8), 2),
                "velocity_mm_h": 0.038,
                "acceleration": 0.015,
                "vibration": 0.025,
                "battery": round(random.uniform(92.0, 99.0), 1),
                "signal_strength": round(random.uniform(-74.0, -66.0), 1),
                "is_offline": False,
            }

    def set_scenario(self, scenario: SimulationScenario):
        """Switches active geotechnical simulation scenario."""
        self.scenario = scenario
        if scenario == SimulationScenario.NORMAL or scenario == SimulationScenario.RECOVERY:
            self.step_count = 0

    def generate_spatial_step(self) -> List[Dict[str, Any]]:
        """
        Advances the simulation by 1 timestep (1 second burst)
        and computes spatially correlated readings for all 9 gallery nodes.
        """
        self.step_count += 1
        now = datetime.now(timezone.utc)
        readings = []

        anchor_state = self.node_states[self.anchor_node_id]

        # -------------------------------------------------------------
        # 1. Update Physical Anchor N5 based on Scenario
        # -------------------------------------------------------------
        if self.scenario == SimulationScenario.NORMAL:
            # Stable baseline with tiny measurement jitter
            anchor_state["velocity_mm_h"] = max(0.035 + random.uniform(-0.005, 0.005), 0.01)
            anchor_state["displacement_mm"] += anchor_state["velocity_mm_h"] * (1.0 / 3600.0) * 8.0
            anchor_state["base_tilt_x"] = 0.85 + random.uniform(-0.02, 0.02)
            anchor_state["base_tilt_y"] = 0.65 + random.uniform(-0.02, 0.02)

        elif self.scenario in (SimulationScenario.DEFORMATION_DEVELOPING, SimulationScenario.RISING_DEFORMATION):
            # Secondary creep: steady increase in deformation rate
            step_factor = min(self.step_count * 0.04, 1.8)
            anchor_state["velocity_mm_h"] = round(0.08 + step_factor * 0.06 + random.uniform(-0.005, 0.005), 4)
            anchor_state["displacement_mm"] += (anchor_state["velocity_mm_h"] * 0.12)
            anchor_state["base_tilt_x"] += 0.04 * (1.0 + step_factor * 0.2)
            anchor_state["base_tilt_y"] += 0.03 * (1.0 + step_factor * 0.2)

        elif self.scenario in (SimulationScenario.HIGH_RISK_ZONE, SimulationScenario.ACCELERATING_DEFORMATION):
            # Tertiary creep: exponential acceleration towards strata collapse
            accel_factor = math.exp(min(self.step_count * 0.09, 3.2))
            anchor_state["velocity_mm_h"] = round(min(0.06 * accel_factor, 1.45), 4)
            anchor_state["displacement_mm"] += min(0.20 * accel_factor, 4.5)
            anchor_state["base_tilt_x"] += min(0.12 * accel_factor, 2.5)
            anchor_state["base_tilt_y"] += min(0.09 * accel_factor, 1.8)
            anchor_state["acceleration"] = round(min(0.03 * accel_factor, 0.5), 3)

        elif self.scenario == SimulationScenario.RECOVERY:
            # Recovery: gradual decay of velocity and stabilization of deformation
            anchor_state["velocity_mm_h"] = max(anchor_state["velocity_mm_h"] * 0.85, 0.038)
            anchor_state["base_tilt_x"] = max(anchor_state["base_tilt_x"] * 0.95, 0.85)
            anchor_state["base_tilt_y"] = max(anchor_state["base_tilt_y"] * 0.95, 0.65)

        # -------------------------------------------------------------
        # 2. Propagate Spatial Field to Simulated Nodes (N1-N4, N6-N9)
        # -------------------------------------------------------------
        decay_rate = 0.85  # Spatial distance attenuation coefficient

        for nid, state in self.node_states.items():
            if nid == self.anchor_node_id:
                # N5 (Real Sensor)
                reading = {
                    "node_id": nid,
                    "timestamp": now.isoformat(),
                    "tilt_x": round(state["base_tilt_x"], 2),
                    "tilt_y": round(state["base_tilt_y"], 2),
                    "acceleration": round(state["acceleration"] + random.uniform(-0.002, 0.002), 3),
                    "displacement_mm": round(state["displacement_mm"], 2),
                    "vibration": round(state["vibration"] + random.uniform(-0.002, 0.002), 3),
                    "battery": round(state["battery"], 1),
                    "signal_strength": round(state["signal_strength"] + random.uniform(-0.8, 0.8), 1),
                    "is_real": True,
                }
                readings.append(reading)
                continue

            # Handle Scenario 5: Communication Failure on peripheral node NODE_09
            if self.scenario == SimulationScenario.COMMUNICATION_FAILURE and nid == "NODE_09":
                state["battery"] = max(state["battery"] - 6.0, 0.0)
                state["signal_strength"] = -125.0
                state["is_offline"] = True
                if self.step_count > 3:
                    continue  # Packet lost / Node offline

            # Handle Scenario 4: Isolated Sensor Fault on NODE_07
            elif self.scenario == SimulationScenario.SENSOR_FAULT and nid == "NODE_07":
                # Dramatic isolated spike on N7 only; anchor N5 and all other nodes remain quiet
                fault_reading = {
                    "node_id": nid,
                    "timestamp": now.isoformat(),
                    "tilt_x": round(state["base_tilt_x"] + 19.5 + random.uniform(-0.5, 0.5), 2),
                    "tilt_y": round(state["base_tilt_y"] + 14.0 + random.uniform(-0.5, 0.5), 2),
                    "acceleration": round(0.52 + random.uniform(-0.02, 0.02), 3),
                    "displacement_mm": round(state["displacement_mm"] + 28.5, 2),
                    "vibration": round(0.35 + random.uniform(0.01, 0.03), 3),
                    "battery": 87.0,
                    "signal_strength": -71.0,
                    "is_real": False,
                }
                readings.append(fault_reading)
                continue

            # Standard Spatially Correlated Propagation from N5
            dist = state["dist_to_anchor"]
            spatial_coupling = math.exp(-decay_rate * dist)  # 1.0 -> ~0.43, 1.414 -> ~0.30

            # Directional bias for realistic mine gallery stress (East-West gallery 2 responds more)
            if nid in ("NODE_04", "NODE_06", "NODE_08", "NODE_02"):
                directional_factor = 1.15
            else:
                directional_factor = 0.85

            effective_coupling = spatial_coupling * directional_factor

            # Correlated velocity & deformation
            anchor_vel_delta = max(anchor_state["velocity_mm_h"] - 0.038, 0.0)
            state["velocity_mm_h"] = round(0.038 + anchor_vel_delta * effective_coupling + random.uniform(-0.003, 0.003), 4)

            anchor_disp_delta = max(anchor_state["displacement_mm"] - 2.0, 0.0)
            state["displacement_mm"] = round(2.0 + anchor_disp_delta * effective_coupling + random.uniform(-0.02, 0.02), 2)

            state["base_tilt_x"] = round(0.75 + (anchor_state["base_tilt_x"] - 0.85) * effective_coupling + random.uniform(-0.01, 0.01), 2)
            state["base_tilt_y"] = round(0.55 + (anchor_state["base_tilt_y"] - 0.65) * effective_coupling + random.uniform(-0.01, 0.01), 2)

            # Battery discharge & signal jitter
            state["battery"] = max(state["battery"] - 0.005, 5.0)

            reading = {
                "node_id": nid,
                "timestamp": now.isoformat(),
                "tilt_x": state["base_tilt_x"],
                "tilt_y": state["base_tilt_y"],
                "acceleration": round(0.02 + (anchor_state["acceleration"] - 0.015) * effective_coupling, 3),
                "displacement_mm": state["displacement_mm"],
                "vibration": round(0.025 + (anchor_state["vibration"] - 0.025) * effective_coupling, 3),
                "battery": round(state["battery"], 1),
                "signal_strength": round(state["signal_strength"] + random.uniform(-1.0, 1.0), 1),
                "is_real": False,
            }
            readings.append(reading)

        return readings


# Global Spatial Field Singleton
spatial_field_simulator = SpatialFieldSimulator()
