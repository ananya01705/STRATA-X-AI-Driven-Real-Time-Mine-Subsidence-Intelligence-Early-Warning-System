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
    DEFORMATION_DEVELOPING = "DEFORMATION_DEVELOPING"
    HIGH_RISK_ZONE = "HIGH_RISK_ZONE"
    RISING_DEFORMATION = "RISING_DEFORMATION"
    ACCELERATING_DEFORMATION = "ACCELERATING_DEFORMATION"
    SENSOR_FAULT = "SENSOR_FAULT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    RECOVERY = "RECOVERY"


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
    is_real: bool = Field(False, description="True if real physical hardware sensor (N5), False if simulated")


class TelemetryResponse(BaseModel):
    accepted: bool
    node_id: str
    timestamp: datetime
    buffer_status: str
    readings_count: int
    required_readings: int = 30
    is_real: bool = False


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
    model_source: Optional[str] = None
    is_real: bool = False


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
    is_real: bool = False


# ==========================================
# LAYER 2 SPATIAL RISK SCHEMAS
# ==========================================

class SpatialRiskZone(BaseModel):
    zone_id: str
    affected_node_ids: List[str]
    number_of_nodes: int
    centroid: Dict[str, float]
    average_deformation: float
    maximum_deformation: float
    average_velocity: float
    maximum_velocity: float
    average_ttf: Optional[float] = None
    minimum_ttf: Optional[float] = None
    risk_level: RiskState
    spatial_confidence: float
    reasons: List[str] = []
    contains_real_sensor: bool = False


class SpatialRiskMapResponse(BaseModel):
    timestamp: datetime
    mine_risk: RiskState
    active_zone_count: int
    zones: List[SpatialRiskZone]
    nodes: List[Dict[str, Any]]
    suppressed_anomalies: List[Dict[str, Any]] = []
    total_nodes: int = 9
    real_sensor_id: str = "NODE_05"


class SpatialNodesResponse(BaseModel):
    timestamp: datetime
    total_nodes: int
    real_nodes_count: int
    simulated_nodes_count: int
    nodes: List[Dict[str, Any]]


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
    is_real: bool = False
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
    system: str = "STRATA-X Underground Mine Subsidence Intelligence System"
    status: str = "operational"
    system_online: bool = True
    mode: str = "OFFLINE_LOCAL_EDGE"
    timestamp: datetime
    total_nodes: int = 9
    active_nodes: int = 9
    online_nodes: int = 9
    critical_nodes: int = 0
    warning_nodes: int = 0
    active_zones_count: int = 0
    overall_mine_risk: RiskState = RiskState.NORMAL
    ml_model_status: str = "READY"
    database_status: str = "CONNECTED"
    real_sensor_node: str = "NODE_05"
    last_sync: Optional[datetime] = None
