from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    HIGH_RISK = "HIGH RISK"
    CRITICAL = "CRITICAL"


class TrendState(str, Enum):
    STABLE = "STABLE"
    DEFORMATION_INCREASING = "DEFORMATION INCREASING"
    ACCELERATION_DETECTED = "ACCELERATION DETECTED"
    HIGH_RISK = "HIGH RISK"
    CRITICAL = "CRITICAL"


class SimulationScenario(str, Enum):
    NORMAL = "NORMAL"
    RISING_DEFORMATION = "RISING_DEFORMATION"
    ACCELERATING_DEFORMATION = "ACCELERATING_DEFORMATION"
    SENSOR_FAULT = "SENSOR_FAULT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"


# ==========================================
# TELEMETRY SCHEMAS
# ==========================================

class SensorReading(BaseModel):
    node_id: str = Field(..., example="NODE_01", description="Unique identifier for sensor node")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of reading")

    tilt_x: float = Field(0.0, description="Tilt angle in X axis (degrees)")
    tilt_y: float = Field(0.0, description="Tilt angle in Y axis (degrees)")
    acceleration: float = Field(0.0, description="Total seismic/vibration acceleration (g or m/s^2)")
    displacement_mm: float = Field(0.0, description="Cumulative roof/strata displacement in mm")
    vibration: float = Field(0.0, description="Peak vibration velocity (mm/s)")
    battery: float = Field(100.0, description="Battery level percentage (0-100)")
    signal_strength: float = Field(-70.0, description="LoRa/Wireless RSSI in dBm")


class TelemetryResponse(BaseModel):
    accepted: bool
    node_id: str
    timestamp: datetime
    buffer_status: str
    readings_count: int
    required_readings: int = 30


# ==========================================
# PREDICTION & RISK SCHEMAS
# ==========================================

class NodePrediction(BaseModel):
    node_id: str
    timestamp: datetime
    status: str  # "READY" | "BUFFERING" | "ERROR"
    buffer_progress: str
    readings_count: int
    ttf_hours: Optional[float] = None
    velocity_mm_h: Optional[float] = None
    deformation_mm: Optional[float] = None
    inverse_velocity: Optional[float] = None


class AnomalyReport(BaseModel):
    is_anomaly: bool
    anomaly_type: Optional[str] = None  # "ISOLATED_SENSOR_FAULT" | "CORRELATED_STRATA_MOVEMENT"
    confidence: float
    description: str


class RiskAssessment(BaseModel):
    node_id: Optional[str] = None
    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Composite Risk index (0-100)")
    risk_state: RiskState
    ttf_hours: Optional[float] = None
    trend: TrendState
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]
    anomaly: Optional[AnomalyReport] = None


# ==========================================
# NODE & SYSTEM HEALTH SCHEMAS
# ==========================================

class NodeInfo(BaseModel):
    node_id: str
    name: str
    grid_x: int
    grid_y: int
    latitude: float
    longitude: float
    depth_m: float
    status: str  # "ONLINE" | "OFFLINE" | "FAULTY" | "DEGRADED"
    battery: float
    signal_strength: float
    last_seen: Optional[datetime] = None
    readings_buffered: int = 0
    current_risk: RiskState = RiskState.NORMAL
    current_ttf_hours: Optional[float] = None


class AlertItem(BaseModel):
    id: str
    timestamp: datetime
    node_id: str
    level: str  # "INFO" | "WARNING" | "CRITICAL"
    title: str
    message: str
    resolved: bool = False


class SystemStatus(BaseModel):
    system: str = "Mine Subsidence Monitoring System"
    status: str = "operational"
    mode: str = "OFFLINE_LOCAL_EDGE"
    timestamp: datetime
    active_nodes_count: int
    online_nodes_count: int
    critical_alerts_count: int
    overall_mine_risk: RiskState
    ml_model_loaded: bool
