try:
    import pytest  # type: ignore
except ImportError:
    pass

from datetime import datetime, timedelta
from backend.services.prediction_service import prediction_service
from backend.services.history_service import history_service


def test_prediction_insufficient_buffer():
    # Only 5 readings appended
    for i in range(5):
        history_service.append_reading({
            "node_id": "NODE_05",
            "tilt_x": 0.8,
            "tilt_y": 0.6,
            "displacement_mm": 2.0 + i * 0.01,
            "vibration": 0.03,
            "timestamp": (datetime.utcnow() + timedelta(seconds=i)).isoformat()
        })

    pred = prediction_service.predict_node_ttf("NODE_05")
    assert pred["status"] == "BUFFERING"
    assert pred["ttf_hours"] is None
    assert pred["readings_count"] == 5
    assert pred["is_real"] is True


def test_prediction_full_buffer_inference():
    # Append 30 readings
    for i in range(30):
        history_service.append_reading({
            "node_id": "NODE_05",
            "tilt_x": 0.8 + i * 0.01,
            "tilt_y": 0.6,
            "displacement_mm": 2.0 + i * 0.05,
            "vibration": 0.03,
            "timestamp": (datetime.utcnow() + timedelta(seconds=i)).isoformat()
        })

    pred = prediction_service.predict_node_ttf("NODE_05")
    assert pred["status"] == "READY"
    assert pred["readings_count"] == 30
    assert pred["ttf_hours"] is not None
    assert pred["ttf_hours"] > 0
    assert pred["model_source"] in ("LSTM_NEURAL_NETWORK", "FUKUZONO_PHYSICS_FALLBACK")
