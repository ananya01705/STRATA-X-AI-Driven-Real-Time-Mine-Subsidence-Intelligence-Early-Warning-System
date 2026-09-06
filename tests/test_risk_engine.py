try:
    import pytest  # type: ignore
except ImportError:
    pass

from backend.core.risk_engine import risk_engine
from backend.models.schemas import RiskState, TrendState


def test_assess_node_risk_normal():
    prediction_result = {
        "ttf_hours": 12.0,
        "velocity_mm_h": 0.035,
        "deformation_mm": 2.0
    }
    assessment = risk_engine.assess_node_risk("NODE_05", prediction_result)
    assert assessment.risk_state == RiskState.NORMAL
    assert assessment.risk_score < 35.0
    assert assessment.is_real is True


def test_assess_node_risk_watch():
    prediction_result = {
        "ttf_hours": 5.0,  # <= TTF_WATCH_HOURS (6.0)
        "velocity_mm_h": 0.08,
        "deformation_mm": 4.5
    }
    assessment = risk_engine.assess_node_risk("NODE_04", prediction_result)
    assert assessment.risk_state in (RiskState.WATCH, RiskState.HIGH_RISK)
    assert assessment.risk_score >= 35.0
    assert assessment.is_real is False


def test_assess_node_risk_high():
    prediction_result = {
        "ttf_hours": 2.5,  # <= TTF_HIGH_RISK_HOURS (3.0)
        "velocity_mm_h": 0.18,
        "deformation_mm": 9.0
    }
    assessment = risk_engine.assess_node_risk("NODE_05", prediction_result)
    assert assessment.risk_state in (RiskState.HIGH_RISK, RiskState.CRITICAL)
    assert assessment.risk_score >= 55.0


def test_assess_node_risk_critical():
    prediction_result = {
        "ttf_hours": 0.8,  # <= TTF_CRITICAL_HOURS (1.0)
        "velocity_mm_h": 0.45,
        "deformation_mm": 25.0
    }
    assessment = risk_engine.assess_node_risk("NODE_05", prediction_result)
    assert assessment.risk_state == RiskState.CRITICAL
    assert assessment.risk_score >= 80.0
    assert assessment.trend == TrendState.CRITICAL


def test_assess_node_risk_sensor_fault_mitigation():
    # Simulate single sensor reporting high velocity while neighbors are calm
    all_readings = {
        "NODE_07": {"velocity": 0.25, "tilt_deg": 18.0},
        "NODE_04": {"velocity": 0.03, "tilt_deg": 1.0},
        "NODE_08": {"velocity": 0.03, "tilt_deg": 1.0},
        "NODE_05": {"velocity": 0.04, "tilt_deg": 1.0},
    }
    prediction_result = {
        "ttf_hours": 8.0,
        "velocity_mm_h": 0.25,
        "deformation_mm": 15.0
    }
    assessment = risk_engine.assess_node_risk("NODE_07", prediction_result, all_readings)
    assert assessment.anomaly is not None
    assert assessment.anomaly.anomaly_type == "ISOLATED_SENSOR_FAULT"
    # Score should be mitigated rather than triggering critical false alarm
    assert assessment.risk_score <= 50.0
