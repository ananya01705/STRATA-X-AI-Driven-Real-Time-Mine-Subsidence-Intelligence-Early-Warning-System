from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ml"
BACKEND_DIR = BASE_DIR / "backend"
STORAGE_DIR = BACKEND_DIR / "storage"

# ML Artifact Paths
ML_MODEL_PATH = ML_DIR / "ttf_lstm_model.keras"
ML_FEATURE_SCALER_PATH = ML_DIR / "lstm_feature_scaler.pkl"
ML_TARGET_SCALER_PATH = ML_DIR / "lstm_target_scaler.pkl"

# Database Configuration
DATABASE_PATH = STORAGE_DIR / "mine_subsidence.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")

# Risk Engine Thresholds (Configurable)
WINDOW_SIZE = 30
TTF_CRITICAL_HOURS = 1.0
TTF_HIGH_RISK_HOURS = 3.0
TTF_WATCH_HOURS = 6.0

# Velocity / Acceleration Thresholds (mm/h and mm/h^2)
VELOCITY_THRESHOLD_HIGH = 0.15  # mm/h
ACCELERATION_THRESHOLD_HIGH = 0.05  # mm/h^2

# Sensor Fault Detection
SENSOR_FAULT_Z_SCORE_THRESHOLD = 3.0
SENSOR_COMM_TIMEOUT_SECONDS = 30.0

# Mine Spatial Grid Configuration (3x3 Grid of Sensor Nodes)
DEFAULT_MINE_NODES = {
    "NODE_01": {"grid_x": 0, "grid_y": 0, "name": "Gallery 1 - West", "lat": 23.6348, "lon": 85.2792, "depth_m": 120.0},
    "NODE_02": {"grid_x": 1, "grid_y": 0, "name": "Gallery 1 - Center", "lat": 23.6348, "lon": 85.2799, "depth_m": 120.0},
    "NODE_03": {"grid_x": 2, "grid_y": 0, "name": "Gallery 1 - East", "lat": 23.6348, "lon": 85.2806, "depth_m": 120.0},
    "NODE_04": {"grid_x": 0, "grid_y": 1, "name": "Gallery 2 - West", "lat": 23.6345, "lon": 85.2792, "depth_m": 125.0},
    "NODE_05": {"grid_x": 1, "grid_y": 1, "name": "Gallery 2 - Center (Junction)", "lat": 23.6345, "lon": 85.2799, "depth_m": 125.0},
    "NODE_06": {"grid_x": 2, "grid_y": 1, "name": "Gallery 2 - East", "lat": 23.6345, "lon": 85.2806, "depth_m": 125.0},
    "NODE_07": {"grid_x": 0, "grid_y": 2, "name": "Gallery 3 - West", "lat": 23.6342, "lon": 85.2792, "depth_m": 130.0},
    "NODE_08": {"grid_x": 1, "grid_y": 2, "name": "Gallery 3 - Center", "lat": 23.6342, "lon": 85.2799, "depth_m": 130.0},
    "NODE_09": {"grid_x": 2, "grid_y": 2, "name": "Gallery 3 - East", "lat": 23.6342, "lon": 85.2806, "depth_m": 130.0},
}
