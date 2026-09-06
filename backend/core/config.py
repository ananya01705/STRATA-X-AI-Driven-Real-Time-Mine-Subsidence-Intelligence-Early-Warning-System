from pathlib import Path
import os
from typing import Dict, Any, List

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    _USE_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore
    _USE_PYDANTIC_SETTINGS = False

from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ml"
BACKEND_DIR = BASE_DIR / "backend"
STORAGE_DIR = BACKEND_DIR / "storage"

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


if _USE_PYDANTIC_SETTINGS:
    class AppSettings(BaseSettings):
        """
        12-Factor Application Settings using Pydantic BaseSettings.
        Loads configurations from environment variables or .env file with type safety.
        """
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )

        # API Server Configuration
        APP_NAME: str = "STRATA-X Mine Subsidence Intelligence System"
        APP_VERSION: str = "2.0.0"
        API_PREFIX: str = ""
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        DEBUG: bool = False
        CORS_ORIGINS: List[str] = ["*"]

        # Database Configuration
        DATABASE_URL: str = Field(
            default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'mine_subsidence.db').as_posix()}")
        )

        # ML Artifact Paths (Layer 1 Ground Truth)
        ML_MODEL_PATH: Path = ML_DIR / "ttf_lstm_model.keras"
        ML_FEATURE_SCALER_PATH: Path = ML_DIR / "lstm_feature_scaler.pkl"
        ML_TARGET_SCALER_PATH: Path = ML_DIR / "lstm_target_scaler.pkl"

        # Risk Engine & TTF Thresholds (Hours)
        WINDOW_SIZE: int = 30
        TTF_CRITICAL_HOURS: float = 1.0
        TTF_HIGH_RISK_HOURS: float = 3.0
        TTF_WATCH_HOURS: float = 6.0

        # Kinematic Thresholds (mm/h and mm/h^2)
        VELOCITY_THRESHOLD_HIGH: float = 0.15
        ACCELERATION_THRESHOLD_HIGH: float = 0.05

        # Sensor Fault Isolation & Watchdog (Seconds)
        SENSOR_FAULT_Z_SCORE_THRESHOLD: float = 3.0
        SENSOR_COMM_TIMEOUT_SECONDS: float = 30.0

        # Hardware Serial Port Configuration (ESP32 / LoRa Bridge)
        SERIAL_PORT: str = "COM3"
        SERIAL_BAUDRATE: int = 115200
        SERIAL_AUTO_RECONNECT: bool = True

        # Real Physical Sensor Identifier
        REAL_SENSOR_NODE_ID: str = "NODE_05"

        # Mine 3x3 Spatial Node Mesh: N5 is Real Hardware, N1-N4 & N6-N9 are Simulated Prototype Nodes
        DEFAULT_MINE_NODES: Dict[str, Dict[str, Any]] = {
            "NODE_01": {"grid_x": 0, "grid_y": 0, "name": "Gallery 1 - West", "lat": 23.6348, "lon": 85.2792, "depth_m": 120.0, "is_real": False},
            "NODE_02": {"grid_x": 1, "grid_y": 0, "name": "Gallery 1 - Center", "lat": 23.6348, "lon": 85.2799, "depth_m": 120.0, "is_real": False},
            "NODE_03": {"grid_x": 2, "grid_y": 0, "name": "Gallery 1 - East", "lat": 23.6348, "lon": 85.2806, "depth_m": 120.0, "is_real": False},
            "NODE_04": {"grid_x": 0, "grid_y": 1, "name": "Gallery 2 - West", "lat": 23.6345, "lon": 85.2792, "depth_m": 125.0, "is_real": False},
            "NODE_05": {"grid_x": 1, "grid_y": 1, "name": "Gallery 2 - Center (Physical Anchor)", "lat": 23.6345, "lon": 85.2799, "depth_m": 125.0, "is_real": True},
            "NODE_06": {"grid_x": 2, "grid_y": 1, "name": "Gallery 2 - East", "lat": 23.6345, "lon": 85.2806, "depth_m": 125.0, "is_real": False},
            "NODE_07": {"grid_x": 0, "grid_y": 2, "name": "Gallery 3 - West", "lat": 23.6342, "lon": 85.2792, "depth_m": 130.0, "is_real": False},
            "NODE_08": {"grid_x": 1, "grid_y": 2, "name": "Gallery 3 - Center", "lat": 23.6342, "lon": 85.2799, "depth_m": 130.0, "is_real": False},
            "NODE_09": {"grid_x": 2, "grid_y": 2, "name": "Gallery 3 - East", "lat": 23.6342, "lon": 85.2806, "depth_m": 130.0, "is_real": False},
        }

else:
    class AppSettings(BaseSettings):
        """Fallback when pydantic_settings is not installed."""
        APP_NAME: str = "STRATA-X Mine Subsidence Intelligence System"
        APP_VERSION: str = "2.0.0"
        API_PREFIX: str = ""
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", 8000))
        DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")
        CORS_ORIGINS: List[str] = ["*"]
        DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'mine_subsidence.db').as_posix()}")

        ML_MODEL_PATH: Path = ML_DIR / "ttf_lstm_model.keras"
        ML_FEATURE_SCALER_PATH: Path = ML_DIR / "lstm_feature_scaler.pkl"
        ML_TARGET_SCALER_PATH: Path = ML_DIR / "lstm_target_scaler.pkl"

        WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", 30))
        TTF_CRITICAL_HOURS: float = float(os.getenv("TTF_CRITICAL_HOURS", 1.0))
        TTF_HIGH_RISK_HOURS: float = float(os.getenv("TTF_HIGH_RISK_HOURS", 3.0))
        TTF_WATCH_HOURS: float = float(os.getenv("TTF_WATCH_HOURS", 6.0))

        VELOCITY_THRESHOLD_HIGH: float = float(os.getenv("VELOCITY_THRESHOLD_HIGH", 0.15))
        ACCELERATION_THRESHOLD_HIGH: float = float(os.getenv("ACCELERATION_THRESHOLD_HIGH", 0.05))

        SENSOR_FAULT_Z_SCORE_THRESHOLD: float = float(os.getenv("SENSOR_FAULT_Z_SCORE_THRESHOLD", 3.0))
        SENSOR_COMM_TIMEOUT_SECONDS: float = float(os.getenv("SENSOR_COMM_TIMEOUT_SECONDS", 30.0))

        SERIAL_PORT: str = os.getenv("SERIAL_PORT", "COM3")
        SERIAL_BAUDRATE: int = int(os.getenv("SERIAL_BAUDRATE", 115200))
        SERIAL_AUTO_RECONNECT: bool = True

        REAL_SENSOR_NODE_ID: str = "NODE_05"

        DEFAULT_MINE_NODES: Dict[str, Dict[str, Any]] = {
            "NODE_01": {"grid_x": 0, "grid_y": 0, "name": "Gallery 1 - West", "lat": 23.6348, "lon": 85.2792, "depth_m": 120.0, "is_real": False},
            "NODE_02": {"grid_x": 1, "grid_y": 0, "name": "Gallery 1 - Center", "lat": 23.6348, "lon": 85.2799, "depth_m": 120.0, "is_real": False},
            "NODE_03": {"grid_x": 2, "grid_y": 0, "name": "Gallery 1 - East", "lat": 23.6348, "lon": 85.2806, "depth_m": 120.0, "is_real": False},
            "NODE_04": {"grid_x": 0, "grid_y": 1, "name": "Gallery 2 - West", "lat": 23.6345, "lon": 85.2792, "depth_m": 125.0, "is_real": False},
            "NODE_05": {"grid_x": 1, "grid_y": 1, "name": "Gallery 2 - Center (Physical Anchor)", "lat": 23.6345, "lon": 85.2799, "depth_m": 125.0, "is_real": True},
            "NODE_06": {"grid_x": 2, "grid_y": 1, "name": "Gallery 2 - East", "lat": 23.6345, "lon": 85.2806, "depth_m": 125.0, "is_real": False},
            "NODE_07": {"grid_x": 0, "grid_y": 2, "name": "Gallery 3 - West", "lat": 23.6342, "lon": 85.2792, "depth_m": 130.0, "is_real": False},
            "NODE_08": {"grid_x": 1, "grid_y": 2, "name": "Gallery 3 - Center", "lat": 23.6342, "lon": 85.2799, "depth_m": 130.0, "is_real": False},
            "NODE_09": {"grid_x": 2, "grid_y": 2, "name": "Gallery 3 - East", "lat": 23.6342, "lon": 85.2806, "depth_m": 130.0, "is_real": False},
        }


# Global Settings Singleton
settings = AppSettings()

# Backward-Compatible Constants Exported for Existing Modules
DATABASE_PATH = STORAGE_DIR / "mine_subsidence.db"
DATABASE_URL = settings.DATABASE_URL
ML_MODEL_PATH = settings.ML_MODEL_PATH
ML_FEATURE_SCALER_PATH = settings.ML_FEATURE_SCALER_PATH
ML_TARGET_SCALER_PATH = settings.ML_TARGET_SCALER_PATH

WINDOW_SIZE = settings.WINDOW_SIZE
TTF_CRITICAL_HOURS = settings.TTF_CRITICAL_HOURS
TTF_HIGH_RISK_HOURS = settings.TTF_HIGH_RISK_HOURS
TTF_WATCH_HOURS = settings.TTF_WATCH_HOURS

VELOCITY_THRESHOLD_HIGH = settings.VELOCITY_THRESHOLD_HIGH
ACCELERATION_THRESHOLD_HIGH = settings.ACCELERATION_THRESHOLD_HIGH

SENSOR_FAULT_Z_SCORE_THRESHOLD = settings.SENSOR_FAULT_Z_SCORE_THRESHOLD
SENSOR_COMM_TIMEOUT_SECONDS = settings.SENSOR_COMM_TIMEOUT_SECONDS
DEFAULT_MINE_NODES = settings.DEFAULT_MINE_NODES
REAL_SENSOR_NODE_ID = settings.REAL_SENSOR_NODE_ID
