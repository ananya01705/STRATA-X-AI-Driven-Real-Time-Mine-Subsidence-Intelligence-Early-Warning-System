import asyncio
import json
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings, SENSOR_COMM_TIMEOUT_SECONDS
from backend.core.exceptions import (
    StrataException,
    strata_exception_handler,
    generic_exception_handler,
)

from backend.api.telemetry import router as telemetry_router
from backend.api.prediction import router as prediction_router
from backend.api.nodes import router as nodes_router
from backend.api.alerts import router as alerts_router
from backend.api.simulation import router as simulation_router
from backend.api.hardware import router as hardware_router
from backend.api.spatial import router as spatial_router

from backend.services.history_service import history_service
from backend.services.prediction_service import prediction_service
from backend.core.risk_engine import risk_engine
from backend.core.spatial_risk import spatial_risk_engine
from backend.models.schemas import SystemStatus, RiskState

logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger("MineSubsidenceBackend")

app = FastAPI(
    title=settings.APP_NAME,
    description="STRATA-X — AI-Driven Real-Time Mine Subsidence Intelligence, Time-to-Failure Prediction & Spatial Risk Early Warning System",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom Exception Handlers
app.add_exception_handler(StrataException, strata_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API Routers
app.include_router(telemetry_router)
app.include_router(prediction_router)
app.include_router(nodes_router)
app.include_router(alerts_router)
app.include_router(simulation_router)
app.include_router(hardware_router)
app.include_router(spatial_router)


# ==========================================
# WEBSOCKET MANAGER FOR REAL-TIME DASHBOARD
# ==========================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)


ws_manager = ConnectionManager()


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming live 1Hz telemetry, node risk assessments,
    and Layer 2 spatial zone detections directly to dashboards.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            latest = history_service.get_latest_reading()
            if latest:
                node_id = latest["node_id"]
                pred = prediction_service.predict_node_ttf(node_id)
                risk = risk_engine.assess_node_risk(node_id, pred)
                
                # Check Layer 2 Spatial Status
                spatial_map = spatial_risk_engine.analyze_spatial_risk()
                
                payload = {
                    "type": "TELEMETRY_UPDATE",
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_id": node_id,
                    "is_real": latest.get("is_real", node_id == "NODE_05"),
                    "tilt_deg": latest.get("tilt_deg", 0.0),
                    "ttf_hours": pred.get("ttf_hours"),
                    "velocity_mm_h": pred.get("velocity_mm_h"),
                    "deformation_mm": pred.get("deformation_mm"),
                    "risk_state": risk.risk_state.value,
                    "risk_score": risk.risk_score,
                    "trend": risk.trend.value,
                    "confidence": risk.confidence,
                    "buffer_status": pred.get("buffer_progress", "READY"),
                    "active_zones_count": spatial_map.active_zone_count,
                    "mine_risk": spatial_map.mine_risk.value,
                }
                await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket streaming error: {e}")
        ws_manager.disconnect(websocket)


# ==========================================
# SYSTEM CORE ENDPOINTS
# ==========================================

@app.get("/", tags=["System"])
def root():
    return {
        "project": "STRATA-X — AI-Driven Mine Subsidence Intelligence",
        "system": "Operational",
        "version": settings.APP_VERSION,
        "real_sensor_anchor": "NODE_05",
        "simulated_nodes": ["NODE_01", "NODE_02", "NODE_03", "NODE_04", "NODE_06", "NODE_07", "NODE_08", "NODE_09"],
        "endpoints": {
            "telemetry": "/telemetry",
            "latest": "/latest",
            "prediction": "/prediction/{node_id}",
            "risk": "/risk",
            "spatial_nodes": "/spatial/nodes",
            "spatial_zones": "/spatial/zones",
            "spatial_risk_map": "/spatial/risk-map",
            "nodes": "/nodes",
            "alerts": "/alerts",
            "events": "/events",
            "hardware": "/hardware/serial/status",
            "simulation": "/simulation/scenario",
            "websocket": "/ws/telemetry",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "OFFLINE_LOCAL_EDGE",
        "ml_engine": "READY" if prediction_service.is_ml_loaded else "FALLBACK_PHYSICS",
        "spatial_engine": "ACTIVE"
    }


@app.get("/system/status", response_model=SystemStatus, tags=["System"])
def get_system_status():
    latest_readings = history_service.get_all_latest_readings()
    from backend.storage.database import db
    active_alerts = db.get_active_alerts()
    crit_count = sum(1 for a in active_alerts if a.get("level") == "CRITICAL")
    warn_count = sum(1 for a in active_alerts if a.get("level") in ("WARNING", "HIGH"))

    # Determine genuinely online nodes based on last_seen timeout
    now = datetime.utcnow()
    online_count = 0
    for nid, r in latest_readings.items():
        ts = r.get("timestamp")
        if ts:
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                if (now - ts_dt).total_seconds() <= SENSOR_COMM_TIMEOUT_SECONDS:
                    online_count += 1
            except Exception:
                online_count += 1
        else:
            online_count += 1

    # Ingest into Layer 2 Spatial Engine
    spatial_map = spatial_risk_engine.analyze_spatial_risk(latest_readings=latest_readings)

    return SystemStatus(
        system="STRATA-X Underground Mine Subsidence Intelligence System",
        status="operational",
        system_online=True,
        mode="OFFLINE_LOCAL_EDGE",
        timestamp=now,
        total_nodes=len(settings.DEFAULT_MINE_NODES),
        active_nodes=len(latest_readings),
        online_nodes=online_count if online_count > 0 else len(latest_readings),
        critical_nodes=crit_count,
        warning_nodes=warn_count,
        active_zones_count=spatial_map.active_zone_count,
        overall_mine_risk=spatial_map.mine_risk,
        ml_model_status="READY" if prediction_service.is_ml_loaded else "FALLBACK_PHYSICS",
        database_status="CONNECTED",
        real_sensor_node=settings.REAL_SENSOR_NODE_ID,
        last_sync=now
    )
