from typing import Dict, Any, Optional
from datetime import datetime
from backend.core.config import SENSOR_COMM_TIMEOUT_SECONDS
from backend.core.spatial import spatial_graph
from backend.models.schemas import AnomalyReport


class HealthEngine:
    """
    Evaluates sensor node health, telemetry reliability, communication timeouts,
    and isolates single sensor faults from genuine geotechnical strata deformation.
    """

    def evaluate_node_health(self, node_data: Dict[str, Any]) -> str:
        """Determines health status: ONLINE, DEGRADED, FAULTY, or OFFLINE."""
        last_seen = node_data.get("timestamp")
        if last_seen:
            if isinstance(last_seen, str):
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen)
                    seconds_ago = (datetime.utcnow() - last_seen_dt).total_seconds()
                    if seconds_ago > SENSOR_COMM_TIMEOUT_SECONDS:
                        return "OFFLINE"
                except Exception:
                    pass

        battery = float(node_data.get("battery", 100.0))
        signal = float(node_data.get("signal_strength", -70.0))

        if battery < 15.0 or signal < -110.0:
            return "DEGRADED"

        return "ONLINE"

    def detect_sensor_fault_vs_movement(
        self,
        target_node_id: str,
        all_node_readings: Dict[str, Dict[str, Any]]
    ) -> AnomalyReport:
        """
        Distinguishes an isolated sensor hardware fault (e.g. fallen sensor, loose mount, noisy ADC)
        from a genuine multi-node strata subsidence event.
        """
        target_reading = all_node_readings.get(target_node_id)
        if not target_reading:
            return AnomalyReport(
                is_anomaly=False,
                confidence=1.0,
                description="No telemetry available for analysis."
            )

        # Collect velocities across all active nodes
        velocities = {
            nid: float(data.get("velocity", 0.0))
            for nid, data in all_node_readings.items()
        }

        corr_score, neighbors, interpretation = spatial_graph.calculate_spatial_correlation(
            target_node_id,
            velocities
        )

        target_vel = velocities.get(target_node_id, 0.0)
        target_tilt = float(target_reading.get("tilt_deg", 0.0))

        # Check for isolated sensor anomaly
        if (target_vel > 0.18 or target_tilt > 12.0) and corr_score < 0.3:
            return AnomalyReport(
                is_anomaly=True,
                anomaly_type="ISOLATED_SENSOR_FAULT",
                confidence=0.88,
                description=(
                    f"ANOMALY DETECTED: Node {target_node_id} reported sudden high movement (velocity: {target_vel:.3f} mm/h, "
                    f"tilt: {target_tilt:.1f}°), but adjacent neighbors ({', '.join(neighbors)}) are completely stable. "
                    "Likely isolated physical sensor disturbance or hardware glitch."
                )
            )

        # Check for correlated genuine ground movement
        if target_vel > 0.10 and corr_score >= 0.8:
            return AnomalyReport(
                is_anomaly=False,
                anomaly_type="CORRELATED_STRATA_MOVEMENT",
                confidence=0.94,
                description=(
                    f"CORRELATED DEFORMATION DETECTED: Consistent strata acceleration verified across "
                    f"neighborhood nodes ({', '.join(neighbors)}). High confidence of authentic subsidence."
                )
            )

        return AnomalyReport(
            is_anomaly=False,
            confidence=0.95,
            description="Sensor telemetry is consistent with background strata baseline."
        )


# Global Singleton
health_engine = HealthEngine()
