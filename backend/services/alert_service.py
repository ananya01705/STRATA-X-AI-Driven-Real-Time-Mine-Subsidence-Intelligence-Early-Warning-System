import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.storage.database import db
from backend.models.schemas import RiskState, RiskAssessment, AlertItem


class AlertService:
    """
    Manages generation, filtering, resolution, and querying of strata hazard alerts.
    """

    def process_node_assessment(self, assessment: RiskAssessment):
        node_id = assessment.node_id or "UNKNOWN"

        # Check for Critical Risk
        if assessment.risk_state == RiskState.CRITICAL:
            alert_id = f"ALERT_CRIT_{node_id}"
            db.insert_alert(
                alert_id=alert_id,
                node_id=node_id,
                level="CRITICAL",
                title=f"🚨 CRITICAL EVACUATION WARNING - {node_id}",
                message=(
                    f"Predicted collapse failure within {assessment.ttf_hours or 0.0:.2f} hours. "
                    f"Composite Risk Score: {assessment.risk_score}/100. "
                    f"Immediate DGMS SCAMP Emergency Protocol activation required."
                )
            )
            db.log_event("CRITICAL_ALERT", f"Critical risk triggered on {node_id}: {assessment.reasons}")

        # Check for High Risk
        elif assessment.risk_state == RiskState.HIGH_RISK:
            alert_id = f"ALERT_HIGH_{node_id}"
            db.insert_alert(
                alert_id=alert_id,
                node_id=node_id,
                level="WARNING",
                title=f"⚠️ HIGH RISK STRATA DISPLACEMENT - {node_id}",
                message=(
                    f"Deformation acceleration detected on {node_id}. "
                    f"TTF: {assessment.ttf_hours or 0.0:.2f} hours. Risk Score: {assessment.risk_score}/100."
                )
            )
            db.log_event("HIGH_RISK_ALERT", f"High risk identified on {node_id}")

        # Check for Anomaly Sensor Fault
        if assessment.anomaly and assessment.anomaly.is_anomaly and assessment.anomaly.anomaly_type == "ISOLATED_SENSOR_FAULT":
            alert_id = f"ALERT_FAULT_{node_id}"
            db.insert_alert(
                alert_id=alert_id,
                node_id=node_id,
                level="INFO",
                title=f"🛠️ SENSOR HARDWARE ANOMALY - {node_id}",
                message=assessment.anomaly.description
            )
            db.log_event("SENSOR_FAULT_DIAGNOSTIC", f"Sensor fault isolated on {node_id}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return db.get_active_alerts()


# Global Singleton
alert_service = AlertService()
