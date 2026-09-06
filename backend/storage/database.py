import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from backend.core.config import DATABASE_PATH, DEFAULT_MINE_NODES

logger = logging.getLogger(__name__)


class Database:
    """
    Offline-First Local SQLite Database for underground mine edge deployment.
    Stores sensor telemetry, historical predictions, alerts, and system audit events.
    """

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema if tables do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Nodes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    name TEXT,
                    grid_x INTEGER,
                    grid_y INTEGER,
                    lat REAL,
                    lon REAL,
                    depth_m REAL,
                    status TEXT,
                    battery REAL,
                    signal_strength REAL,
                    last_seen TEXT
                )
            """)

            # Telemetry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    timestamp TEXT,
                    tilt_x REAL,
                    tilt_y REAL,
                    acceleration REAL,
                    displacement_mm REAL,
                    vibration REAL,
                    battery REAL,
                    signal_strength REAL,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                )
            """)

            # Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    timestamp TEXT,
                    ttf_hours REAL,
                    velocity_mm_h REAL,
                    deformation_mm REAL,
                    risk_score REAL,
                    risk_state TEXT,
                    trend TEXT,
                    confidence REAL,
                    reasons TEXT,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                )
            """)

            # Alerts Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    node_id TEXT,
                    level TEXT,
                    title TEXT,
                    message TEXT,
                    resolved INTEGER DEFAULT 0
                )
            """)

            # Events Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    details TEXT
                )
            """)

            # Seed default nodes if empty
            cursor.execute("SELECT COUNT(*) FROM nodes")
            if cursor.fetchone()[0] == 0:
                for nid, ninfo in DEFAULT_MINE_NODES.items():
                    cursor.execute("""
                        INSERT INTO nodes (node_id, name, grid_x, grid_y, lat, lon, depth_m, status, battery, signal_strength, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        nid,
                        ninfo["name"],
                        ninfo["grid_x"],
                        ninfo["grid_y"],
                        ninfo["lat"],
                        ninfo["lon"],
                        ninfo["depth_m"],
                        "ONLINE",
                        100.0,
                        -70.0,
                        datetime.utcnow().isoformat()
                    ))
            
            conn.commit()

    def record_telemetry(self, reading: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            ts = reading.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            elif not ts:
                ts = datetime.utcnow().isoformat()

            cursor.execute("""
                INSERT INTO telemetry (node_id, timestamp, tilt_x, tilt_y, acceleration, displacement_mm, vibration, battery, signal_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading["node_id"],
                ts,
                reading.get("tilt_x", 0.0),
                reading.get("tilt_y", 0.0),
                reading.get("acceleration", 0.0),
                reading.get("displacement_mm", 0.0),
                reading.get("vibration", 0.0),
                reading.get("battery", 100.0),
                reading.get("signal_strength", -70.0)
            ))

            cursor.execute("""
                UPDATE nodes 
                SET last_seen = ?, battery = ?, signal_strength = ?
                WHERE node_id = ?
            """, (ts, reading.get("battery", 100.0), reading.get("signal_strength", -70.0), reading["node_id"]))
            conn.commit()

    def record_prediction(self, pred: Dict[str, Any], risk: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (node_id, timestamp, ttf_hours, velocity_mm_h, deformation_mm, risk_score, risk_state, trend, confidence, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred["node_id"],
                pred.get("timestamp", datetime.utcnow().isoformat()),
                pred.get("ttf_hours"),
                pred.get("velocity_mm_h"),
                pred.get("deformation_mm"),
                risk.get("risk_score", 0.0),
                risk.get("risk_state", "NORMAL"),
                risk.get("trend", "STABLE"),
                risk.get("confidence", 1.0),
                json.dumps(risk.get("reasons", []))
            ))
            conn.commit()

    def insert_alert(self, alert_id: str, node_id: str, level: str, title: str, message: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts (id, timestamp, node_id, level, title, message, resolved)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (alert_id, datetime.utcnow().isoformat(), node_id, level, title, message))
            conn.commit()

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC LIMIT 20")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def log_event(self, event_type: str, details: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (timestamp, event_type, details)
                VALUES (?, ?, ?)
            """, (datetime.utcnow().isoformat(), event_type, details))
            conn.commit()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes ORDER BY node_id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def get_node_telemetry_history(self, node_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM telemetry 
                WHERE node_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (node_id, limit))
            return [dict(r) for r in cursor.fetchall()]


# Global Singleton
db = Database()
