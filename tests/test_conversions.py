import math
from datetime import datetime, timedelta
from backend.core.conversions import compute_resultant_tilt, transform_reading_to_ml_format

try:
    import pytest  # type: ignore
    approx = pytest.approx
except ImportError:
    def approx(val, rel=1e-3, abs=1e-3):
        class Approx:
            def __init__(self, v): self.v = v
            def __eq__(self, other): return math.isclose(other, self.v, rel_tol=rel, abs_tol=abs)
            def __repr__(self): return f"approx({self.v})"
        return Approx(val)


def test_compute_resultant_tilt():
    # 3-4-5 triangle
    assert compute_resultant_tilt(3.0, 4.0) == approx(5.0, abs=1e-4)
    # Zero tilt
    assert compute_resultant_tilt(0.0, 0.0) == approx(0.0, abs=1e-4)
    # Single axis
    assert compute_resultant_tilt(2.5, 0.0) == approx(2.5, abs=1e-4)


def test_transform_reading_to_ml_format_initial():
    reading = {
        "node_id": "NODE_05",
        "tilt_x": 1.0,
        "tilt_y": 1.0,
        "displacement_mm": 5.0,
        "vibration": 0.04,
        "timestamp": datetime.utcnow().isoformat()
    }
    features = transform_reading_to_ml_format(reading, previous_reading=None)
    
    assert features["deformation"] == approx(5.0, abs=0.01)
    assert features["velocity"] >= 0.001
    assert features["inverse_velocity"] == approx(1.0 / features["velocity"], abs=0.001)
    assert features["tilt_deg"] == approx(math.sqrt(2.0), abs=0.01)


def test_transform_reading_to_ml_format_with_previous():
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=36)  # 0.01 hours

    prev_reading = {
        "deformation": 5.0,
        "timestamp": t0.isoformat()
    }
    curr_reading = {
        "displacement_mm": 5.5,  # 0.5 mm change in 0.01 h -> 50 mm/h
        "tilt_x": 1.0,
        "tilt_y": 0.0,
        "timestamp": t1.isoformat()
    }

    features = transform_reading_to_ml_format(curr_reading, prev_reading)
    assert features["deformation"] == approx(5.5, abs=0.01)
    assert features["velocity"] == approx(50.0, abs=0.5)
    assert features["inverse_velocity"] == approx(1.0 / 50.0, abs=0.001)


def test_transform_reading_zero_division_safeguard():
    # If deformation didn't change, velocity should default to non-zero positive minimum
    t0 = datetime.utcnow()
    t1 = t0 + timedelta(seconds=10)

    prev_reading = {
        "deformation": 3.0,
        "timestamp": t0.isoformat()
    }
    curr_reading = {
        "displacement_mm": 3.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
        "timestamp": t1.isoformat()
    }

    features = transform_reading_to_ml_format(curr_reading, prev_reading)
    assert features["velocity"] >= 0.001
    assert not math.isinf(features["inverse_velocity"])
    assert not math.isnan(features["inverse_velocity"])
