from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Dict, Any

from simulator.mine_simulator import simulator
from backend.models.schemas import SimulationScenario

router = APIRouter(prefix="/simulation", tags=["Simulation Controls"])


class ScenarioRequest(BaseModel):
    scenario: SimulationScenario
    interval_seconds: float = 1.5


@router.post("/scenario")
def set_simulation_scenario(req: ScenarioRequest):
    """Changes the active simulation scenario dynamically (NORMAL, RISING_DEFORMATION, ACCELERATING_DEFORMATION, SENSOR_FAULT, COMMUNICATION_FAILURE)."""
    simulator.set_scenario(req.scenario)
    return {
        "status": "scenario_updated",
        "active_scenario": req.scenario.value,
        "is_simulator_running": simulator.is_running
    }


@router.post("/start")
def start_simulator(interval_seconds: float = Query(1.5, description="Seconds between telemetry bursts")):
    """Starts background automated telemetry generation for all 9 sensor nodes."""
    simulator.start(interval_seconds=interval_seconds)
    return {
        "status": "simulator_running",
        "interval_seconds": interval_seconds,
        "active_scenario": simulator.scenario.value
    }


@router.post("/stop")
def stop_simulator():
    """Stops background telemetry generation."""
    simulator.stop()
    return {
        "status": "simulator_stopped"
    }


@router.get("/status")
def get_simulation_status():
    """Returns whether the simulator is currently generating telemetry and the active scenario."""
    return {
        "is_running": simulator.is_running,
        "active_scenario": simulator.scenario.value,
        "step_count": simulator.step_count
    }
