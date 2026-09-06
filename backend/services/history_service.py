from collections import deque
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
from backend.core.config import WINDOW_SIZE
from backend.core.conversions import transform_reading_to_ml_format


class HistoryService:
    """
    Maintains independent rolling history buffers (up to WINDOW_SIZE=30) per sensor node.
    Guarantees that telemetry from different nodes are never mixed.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._node_buffers: Dict[str, deque] = {}
        self._last_raw_readings: Dict[str, Dict[str, Any]] = {}

    def append_reading(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends a new raw telemetry reading for a node, converting it to ML format
        and updating the node's rolling window.
        """
        node_id = reading["node_id"]
        if node_id not in self._node_buffers:
            self._node_buffers[node_id] = deque(maxlen=self.window_size)

        prev_reading = self._last_raw_readings.get(node_id)
        
        # Convert raw reading to ML format features
        ml_features = transform_reading_to_ml_format(reading, prev_reading)
        
        record = {
            "node_id": node_id,
            "timestamp": reading.get("timestamp", datetime.utcnow().isoformat()),
            "tilt_x": reading.get("tilt_x", 0.0),
            "tilt_y": reading.get("tilt_y", 0.0),
            "acceleration": reading.get("acceleration", 0.0),
            "displacement_mm": reading.get("displacement_mm", 0.0),
            "vibration": reading.get("vibration", 0.0),
            "battery": reading.get("battery", 100.0),
            "signal_strength": reading.get("signal_strength", -70.0),
            "tilt_deg": ml_features["tilt_deg"],
            "deformation": ml_features["deformation"],
            "velocity": ml_features["velocity"],
            "inverse_velocity": ml_features["inverse_velocity"],
        }

        self._node_buffers[node_id].append(record)
        self._last_raw_readings[node_id] = record

        buffer_len = len(self._node_buffers[node_id])
        return {
            "node_id": node_id,
            "readings_count": buffer_len,
            "required_readings": self.window_size,
            "buffer_status": "READY" if buffer_len >= self.window_size else f"BUFFERING ({buffer_len}/{self.window_size})"
        }

    def get_node_history_df(self, node_id: str) -> Optional[pd.DataFrame]:
        """Returns the recent history for a node as a pandas DataFrame."""
        if node_id not in self._node_buffers or len(self._node_buffers[node_id]) == 0:
            return None
        return pd.DataFrame(list(self._node_buffers[node_id]))

    def get_node_buffer_count(self, node_id: str) -> int:
        return len(self._node_buffers.get(node_id, []))

    def has_sufficient_data(self, node_id: str) -> bool:
        return self.get_node_buffer_count(node_id) >= self.window_size

    def get_latest_reading(self, node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the most recent reading for a given node or the latest across all nodes."""
        if node_id:
            return self._last_raw_readings.get(node_id)
        if not self._last_raw_readings:
            return None
        # Return most recently updated node reading
        return max(self._last_raw_readings.values(), key=lambda r: str(r.get("timestamp", "")))

    def get_all_latest_readings(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._last_raw_readings)

    def clear_node_history(self, node_id: str):
        if node_id in self._node_buffers:
            self._node_buffers[node_id].clear()
        if node_id in self._last_raw_readings:
            del self._last_raw_readings[node_id]


# Global Singleton
history_service = HistoryService()
