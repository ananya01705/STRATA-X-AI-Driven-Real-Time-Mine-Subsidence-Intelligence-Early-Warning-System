import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable

try:
    import serial
    import serial.tools.list_ports
    _serial_available = True
except ImportError:
    _serial_available = False

from backend.core.config import settings
from backend.services.history_service import history_service
from backend.core.conversions import compute_kinematics

logger = logging.getLogger(__name__)


class SerialIngestionService:
    """
    Background worker service that reads live JSON telemetry from an ESP32 / LoRa gateway
    over USB COM Serial Port (115200 baud) and automatically forwards it to the history service.
    """

    def __init__(self):
        self.port: str = settings.SERIAL_PORT
        self.baudrate: int = settings.SERIAL_BAUDRATE
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.serial_conn = None
        self.last_packet_time: Optional[datetime] = None
        self.packet_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.on_packet_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def list_available_ports(self):
        if not _serial_available:
            return []
        ports = serial.tools.list_ports.comports()
        return [{"port": p.device, "description": p.description, "hwid": p.hwid} for p in ports]

    def start(self, port: Optional[str] = None, baudrate: Optional[int] = None):
        if not _serial_available:
            logger.warning("pyserial is not installed. Hardware serial listener disabled.")
            return False

        if self.is_running:
            logger.info("Serial listener is already running.")
            return True

        if port:
            self.port = port
        if baudrate:
            self.baudrate = baudrate

        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"Serial Ingestion Service started on port {self.port} at {self.baudrate} baud.")
        return True

    def stop(self):
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("Serial Ingestion Service stopped.")

    def _read_loop(self):
        while self.is_running:
            try:
                logger.info(f"Attempting connection to Serial Port: {self.port}...")
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
                logger.info(f"Connected to ESP32 Hardware on {self.port}.")

                while self.is_running and self.serial_conn.is_open:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    # Parse JSON stream from ESP32 (e.g. {"node_id": 1, "tilt_x": 0.0, "tilt_y": 0.0, "accel_z": 9.8})
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            node_id_raw = data.get("node_id", 1)
                            node_id = f"NODE_{int(node_id_raw):02d}" if isinstance(node_id_raw, (int, str)) and str(node_id_raw).isdigit() else str(node_id_raw)
                            
                            payload = {
                                "node_id": node_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "tilt_x": float(data.get("tilt_x", 0.0)),
                                "tilt_y": float(data.get("tilt_y", 0.0)),
                                "acceleration": float(data.get("accel_z", data.get("acceleration", 0.0))),
                                "displacement_mm": float(data.get("displacement_mm", data.get("tilt_x", 0.0) * 5.0)),
                                "vibration": float(data.get("vibration", 0.02)),
                                "battery": float(data.get("battery", 98.0)),
                                "signal_strength": float(data.get("signal_strength", -65.0))
                            }

                            kinematics = compute_kinematics(node_id, payload)
                            history_service.add_reading(node_id, kinematics)
                            self.packet_count += 1
                            self.last_packet_time = datetime.utcnow()

                            if self.on_packet_callback:
                                self.on_packet_callback(kinematics)

                        except Exception as parse_err:
                            self.error_count += 1
                            self.last_error = f"JSON parse error: {parse_err}"
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.warning(f"Serial port '{self.port}' connection waiting/retry: {e}")
                time.sleep(3.0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "port": self.port,
            "baudrate": self.baudrate,
            "packet_count": self.packet_count,
            "error_count": self.error_count,
            "last_packet_time": self.last_packet_time.isoformat() if self.last_packet_time else None,
            "last_error": self.last_error,
            "available_ports": self.list_available_ports()
        }


serial_service = SerialIngestionService()
