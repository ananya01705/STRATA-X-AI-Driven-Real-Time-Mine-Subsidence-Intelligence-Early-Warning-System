import time
import math
import random
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests

from backend.core.config import DEFAULT_MINE_NODES
from backend.models.schemas import SimulationScenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MineSimulator")


class MineSimulator:
    """
    Realistic Underground Mine Sensor Network Simulator.
    Simulates multi-node wireless telemetry across 5 geotechnical and operational scenarios:
    1. NORMAL (Stable background strata)
    2. RISING_DEFORMATION (Secondary creep strain accumulation)
    3. ACCELERATING_DEFORMATION (Tertiary creep progression toward collapse)
    4. SENSOR_FAULT (Isolated hardware anomaly with stable surroundings)
    5. COMMUNICATION_FAILURE (Node dropout / packet loss)
    """

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.scenario: SimulationScenario = SimulationScenario.NORMAL
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.step_count = 0
        self.node_states: Dict[str, Dict[str, Any]] = {}
        self.init_nodes()

    def init_nodes(self):
        self.step_count = 0
        self.node_states = {}
        for nid in DEFAULT_MINE_NODES.keys():
            self.node_states[nid] = {
                "base_tilt_x": round(random.uniform(0.5, 1.2), 2),
                "base_tilt_y": round(random.uniform(0.3, 0.9), 2),
                "displacement_mm": round(random.uniform(1.0, 3.0), 2),
                "velocity": 0.04,
                "battery": round(random.uniform(90.0, 99.0), 1),
                "signal_strength": round(random.uniform(-75.0, -65.0), 1),
                "offline": False
            }

    def set_scenario(self, scenario: SimulationScenario):
        self.scenario = scenario
        logger.info(f"Simulator scenario set to: {scenario.value}")
        if scenario == SimulationScenario.NORMAL:
            self.init_nodes()

    def generate_step_readings(self) -> List[Dict[str, Any]]:
        self.step_count += 1
        now = datetime.utcnow()
        readings = []

        # Target cluster for deformation scenarios (Gallery 2 center)
        cluster_nodes = ["NODE_04", "NODE_05", "NODE_06"]

        for nid in DEFAULT_MINE_NODES.keys():
            state = self.node_states[nid]

            # Scenario 5: Communication Failure on NODE_09
            if self.scenario == SimulationScenario.COMMUNICATION_FAILURE and nid == "NODE_09":
                state["battery"] = max(state["battery"] - 5.0, 0.0)
                state["signal_strength"] = -120.0
                if self.step_count > 3:
                    continue  # Packet dropped / node offline

            # Scenario 4: Isolated Sensor Fault on NODE_07
            elif self.scenario == SimulationScenario.SENSOR_FAULT and nid == "NODE_07":
                # Spike isolated sensor reading dramatically
                reading = {
                    "node_id": nid,
                    "timestamp": now.isoformat(),
                    "tilt_x": round(state["base_tilt_x"] + 18.5 + random.uniform(-1.0, 1.0), 2),
                    "tilt_y": round(state["base_tilt_y"] + 12.0 + random.uniform(-1.0, 1.0), 2),
                    "acceleration": round(0.45 + random.uniform(-0.05, 0.05), 3),
                    "displacement_mm": round(state["displacement_mm"] + 25.0, 2),
                    "vibration": round(0.28 + random.uniform(0.01, 0.05), 3),
                    "battery": 88.0,
                    "signal_strength": -72.0
                }
                readings.append(reading)
                continue

            # Scenario 2: Rising Deformation on Gallery 2 cluster
            elif self.scenario == SimulationScenario.RISING_DEFORMATION:
                if nid in cluster_nodes:
                    state["displacement_mm"] += 0.15 + random.uniform(0.02, 0.05)
                    state["base_tilt_x"] += 0.08
                    state["velocity"] = 0.12
                else:
                    state["displacement_mm"] += random.uniform(0.001, 0.005)

            # Scenario 3: Accelerating Deformation (Tertiary Creep toward failure)
            elif self.scenario == SimulationScenario.ACCELERATING_DEFORMATION:
                # Exponential acceleration: displacement accelerates with step_count
                accel_factor = math.exp(min(self.step_count * 0.08, 3.5))
                if nid in cluster_nodes:
                    state["displacement_mm"] += 0.25 * accel_factor
                    state["base_tilt_x"] += 0.15 * accel_factor
                    state["base_tilt_y"] += 0.10 * accel_factor
                    state["velocity"] = min(0.05 * accel_factor, 1.5)
                else:
                    # Neighboring galleries also experience sympathetic stress
                    state["displacement_mm"] += 0.05 * (accel_factor * 0.4)
                    state["base_tilt_x"] += 0.03 * (accel_factor * 0.3)

            # Scenario 1: Normal baseline
            else:
                noise = random.uniform(-0.01, 0.01)
                state["displacement_mm"] += max(0.002 + noise, 0.0)

            # Natural battery discharge & signal jitter
            state["battery"] = max(state["battery"] - 0.01, 5.0)
            signal_jitter = random.uniform(-1.5, 1.5)

            reading = {
                "node_id": nid,
                "timestamp": now.isoformat(),
                "tilt_x": round(state["base_tilt_x"] + random.uniform(-0.02, 0.02), 2),
                "tilt_y": round(state["base_tilt_y"] + random.uniform(-0.02, 0.02), 2),
                "acceleration": round(0.02 + random.uniform(-0.005, 0.005), 3),
                "displacement_mm": round(state["displacement_mm"], 2),
                "vibration": round(0.03 + random.uniform(-0.005, 0.005), 3),
                "battery": round(state["battery"], 1),
                "signal_strength": round(state["signal_strength"] + signal_jitter, 1)
            }
            readings.append(reading)

        return readings

    def run_loop(self, interval_seconds: float = 2.0):
        logger.info("Simulator loop started...")
        while self.is_running:
            readings = self.generate_step_readings()
            for r in readings:
                try:
                    resp = requests.post(f"{self.backend_url}/telemetry", json=r, timeout=2.0)
                except Exception as e:
                    logger.debug(f"Failed to post reading for {r['node_id']}: {e}")
            time.sleep(interval_seconds)

    def start(self, interval_seconds: float = 2.0):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self.run_loop, args=(interval_seconds,), daemon=True)
        self._thread.start()
        logger.info("Mine simulator started in background.")

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Mine simulator stopped.")


# Global Singleton
simulator = MineSimulator()


if __name__ == "__main__":
    import sys
    print("==================================================")
    print("SIH 26025 — Underground Mine Subsidence Simulator")
    print("==================================================")
    sim = MineSimulator()
    sim.set_scenario(SimulationScenario.NORMAL)
    sim.start(interval_seconds=1.5)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim.stop()
        print("\nSimulator stopped.")
