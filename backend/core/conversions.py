import numpy as np
import math
from typing import Dict, Any, Optional
from datetime import datetime


def compute_resultant_tilt(tilt_x: float, tilt_y: float) -> float:
    """Calculate resultant tilt magnitude in degrees: sqrt(tilt_x^2 + tilt_y^2)."""
    return float(math.sqrt(tilt_x ** 2 + tilt_y ** 2))


def transform_reading_to_ml_format(
    current_reading: Dict[str, Any],
    previous_reading: Optional[Dict[str, Any]] = None,
    dt_hours_fallback: float = 0.01
) -> Dict[str, float]:
    """
    Transforms raw telemetry sensor data into ML-compatible features.
    
    ML expected features:
    - deformation (cumulative movement/displacement in mm or tilt-derived unit)
    - velocity (rate of deformation change in unit/hour)
    - inverse_velocity (1 / velocity with zero-protection)
    """
    # 1. Total cumulative deformation
    displacement = float(current_reading.get("displacement_mm", 0.0))
    tilt_mag = compute_resultant_tilt(
        float(current_reading.get("tilt_x", 0.0)),
        float(current_reading.get("tilt_y", 0.0))
    )
    
    # Combined effective deformation index (mm)
    deformation = displacement if displacement > 0 else (tilt_mag * 5.0)

    # 2. Velocity calculation (rate of deformation change)
    if previous_reading is not None:
        try:
            curr_time = current_reading.get("timestamp")
            prev_time = previous_reading.get("timestamp")
            if isinstance(curr_time, str):
                curr_time = datetime.fromisoformat(curr_time.replace("Z", "+00:00")).replace(tzinfo=None)
            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time.replace("Z", "+00:00")).replace(tzinfo=None)
            dt_seconds = max((curr_time - prev_time).total_seconds(), 0.1)
            dt_hours = dt_seconds / 3600.0
        except Exception:
            dt_hours = dt_hours_fallback

        prev_def = float(previous_reading.get("deformation", deformation))
        diff = max(deformation - prev_def, 0.0)
        velocity = diff / max(dt_hours, 1e-5)
    else:
        # Initial estimate based on vibration / tilt rate baseline
        velocity = max(float(current_reading.get("vibration", 0.05)), 0.035)

    # Ensure non-negative & non-zero velocity
    velocity = max(velocity, 0.001)

    # 3. Inverse velocity with zero protection
    inverse_velocity = 1.0 / velocity

    return {
        "deformation": deformation,
        "velocity": velocity,
        "inverse_velocity": inverse_velocity,
        "tilt_deg": tilt_mag,
    }


def compute_kinematics(node_id: str, reading: Dict[str, Any]) -> Dict[str, Any]:
    """Helper for hardware ingestion to transform raw sensor packet with kinematic features."""
    ml_format = transform_reading_to_ml_format(reading)
    return {
        **reading,
        "node_id": node_id,
        "tilt_deg": ml_format["tilt_deg"],
        "deformation": ml_format["deformation"],
        "velocity": ml_format["velocity"],
        "inverse_velocity": ml_format["inverse_velocity"],
    }
