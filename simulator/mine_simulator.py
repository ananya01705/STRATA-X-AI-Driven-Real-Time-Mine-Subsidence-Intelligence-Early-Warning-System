import time
import threading
import logging
from typing import Dict, Any, List, Optional
import requests

from backend.models.schemas import SimulationScenario
from simulator.spatial_field import spatial_field_simulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MineSimulator")


class MineSimulator:
    """
    Realistic Underground Mine Sensor Network Simulator.
    Simulates multi-node wireless telemetry across geotechnical and operational scenarios:
    1. NORMAL (Stable background strata)
    2. DEFORMATION_DEVELOPING / RISING_DEFORMATION (Secondary creep strain accumulation)
    3. HIGH_RISK_ZONE / ACCELERATING_DEFORMATION (Tertiary creep progression toward collapse)
    4. SENSOR_FAULT (Isolated hardware anomaly with stable surroundings)
    5. COMMUNICATION_FAILURE (Node dropout / packet loss)
    6. RECOVERY (Decaying deformation back to stable baseline)
    """

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.scenario: SimulationScenario = SimulationScenario.NORMAL
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.field = spatial_field_simulator

    @property
    def step_count(self) -> int:
        return self.field.step_count

    def set_scenario(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.field.set_scenario(scenario)
        logger.info(f"Simulator scenario set to: {scenario.value}")

    def generate_step_readings(self) -> List[Dict[str, Any]]:
        return self.field.generate_spatial_step()

    def run_loop(self, interval_seconds: float = 1.5):
        logger.info("Simulator loop started...")
        while self.is_running:
            readings = self.generate_step_readings()
            for r in readings:
                try:
                    requests.post(f"{self.backend_url}/telemetry", json=r, timeout=2.0)
                except Exception as e:
                    logger.debug(f"Failed to post reading for {r['node_id']}: {e}")
            time.sleep(interval_seconds)

    def start(self, interval_seconds: float = 1.5):
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
    print("==================================================")
    print("STRATA-X — Underground Mine Subsidence Simulator")
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
