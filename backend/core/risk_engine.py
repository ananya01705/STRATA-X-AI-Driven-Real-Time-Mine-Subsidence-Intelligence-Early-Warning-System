from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from backend.core.config import (
    TTF_CRITICAL_HOURS,
    TTF_HIGH_RISK_HOURS,
    TTF_WATCH_HOURS,
    VELOCITY_THRESHOLD_HIGH
)
from backend.models.schemas import RiskState, TrendState, RiskAssessment
from backend.core.health import health_engine
from backend.services.history_service import history_service


class MultiFactorRiskEngine:
    """
    Unified Geotechnical Risk & Sensor Fusion Engine.
    Combines Member 1's LSTM TTF prediction with deformation rate,
    acceleration slope, temporal trend state, spatial correlation, and sensor fault intelligence.
    """

    def assess_node_risk(
        self,
        node_id: str,
        prediction_result: Dict[str, Any],
        all_latest_readings: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> RiskAssessment:
        reasons: List[str] = []
        now = datetime.utcnow()

        ttf = prediction_result.get("ttf_hours")
        velocity = prediction_result.get("velocity_mm_h", 0.0) or 0.0
        deformation = prediction_result.get("deformation_mm", 0.0) or 0.0

        # 1. Evaluate temporal trend & acceleration from node history
        trend = TrendState.STABLE
        df = history_service.get_node_history_df(node_id)
        
        accel = 0.0
        if df is not None and len(df) >= 10:
            vels = df["velocity"].tail(10).values
            if len(vels) >= 2:
                accel = float(vels[-1] - vels[0])

        if ttf is not None and ttf <= TTF_CRITICAL_HOURS:
            trend = TrendState.CRITICAL
            reasons.append(f"CRITICAL TTF: Predicted failure in {ttf:.2f} hours (<= {TTF_CRITICAL_HOURS}h threshold).")
        elif ttf is not None and ttf <= TTF_HIGH_RISK_HOURS:
            trend = TrendState.HIGH_RISK
            reasons.append(f"HIGH RISK TTF: Failure predicted in {ttf:.2f} hours.")
        elif accel > 0.05:
            trend = TrendState.ACCELERATION_DETECTED
            reasons.append(f"Tertiary Creep Warning: Positive deformation acceleration detected (+{accel:.3f} mm/h).")
        elif velocity > VELOCITY_THRESHOLD_HIGH:
            trend = TrendState.DEFORMATION_INCREASING
            reasons.append(f"Elevated deformation rate: {velocity:.3f} mm/h exceeds baseline.")
        else:
            trend = TrendState.STABLE
            reasons.append("Strata conditions stable within operational safety envelope.")

        # 2. Sensor Fault & Spatial Analysis
        if all_latest_readings is None:
            all_latest_readings = history_service.get_all_latest_readings()

        anomaly_report = health_engine.detect_sensor_fault_vs_movement(
            node_id,
            all_latest_readings
        )

        if anomaly_report.is_anomaly and anomaly_report.anomaly_type == "ISOLATED_SENSOR_FAULT":
            reasons.append("⚠️ Isolated Sensor Discrepancy: Movement not confirmed by adjacent gallery nodes.")

        if anomaly_report.anomaly_type == "CORRELATED_STRATA_MOVEMENT":
            reasons.append("🚨 Spatial Confirmation: Multiple adjacent nodes show synchronized strata deformation.")

        # 3. Multi-Factor Composite Risk Score (0 - 100)
        risk_score = 10.0  # baseline normal

        # TTF Contribution (0 - 50 pts)
        if ttf is not None:
            if ttf <= TTF_CRITICAL_HOURS:
                risk_score += 50.0
            elif ttf <= TTF_HIGH_RISK_HOURS:
                risk_score += 35.0 + (TTF_HIGH_RISK_HOURS - ttf) * 7.5
            elif ttf <= TTF_WATCH_HOURS:
                risk_score += 15.0 + (TTF_WATCH_HOURS - ttf) * 6.6
            else:
                risk_score += max(15.0 - (ttf - TTF_WATCH_HOURS), 0.0)

        # Velocity Contribution (0 - 30 pts)
        vel_score = min((velocity / 0.20) * 30.0, 30.0)
        risk_score += vel_score

        # Acceleration Contribution (0 - 20 pts)
        if accel > 0:
            risk_score += min((accel / 0.10) * 20.0, 20.0)

        # Sensor Fault Mitigation: If an isolated sensor fault is flagged, down-weight score
        confidence = 0.92
        if anomaly_report.is_anomaly and anomaly_report.anomaly_type == "ISOLATED_SENSOR_FAULT":
            risk_score = min(risk_score * 0.45, 45.0)  # De-escalate false alarm
            confidence = anomaly_report.confidence

        risk_score = round(min(max(risk_score, 0.0), 100.0), 1)

        # 4. Final Risk State Mapping (Preserving Member 1's Baseline)
        if risk_score >= 80.0 or (ttf is not None and ttf <= TTF_CRITICAL_HOURS):
            risk_state = RiskState.CRITICAL
        elif risk_score >= 55.0 or (ttf is not None and ttf <= TTF_HIGH_RISK_HOURS):
            risk_state = RiskState.HIGH_RISK
        elif risk_score >= 35.0 or (ttf is not None and ttf <= TTF_WATCH_HOURS):
            risk_state = RiskState.WATCH
        else:
            risk_state = RiskState.NORMAL

        return RiskAssessment(
            node_id=node_id,
            timestamp=now,
            risk_score=risk_score,
            risk_state=risk_state,
            ttf_hours=ttf,
            trend=trend,
            confidence=confidence,
            reasons=reasons,
            anomaly=anomaly_report
        )


# Global Singleton
risk_engine = MultiFactorRiskEngine()
