import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from backend.api.telemetry import router as telemetry_router
from backend.api.prediction import router as prediction_router
from backend.api.nodes import router as nodes_router
from backend.api.alerts import router as alerts_router
from backend.api.simulation import router as simulation_router

from backend.services.history_service import history_service
from backend.services.prediction_service import prediction_service
from backend.core.risk_engine import risk_engine
from backend.models.schemas import SystemStatus, RiskState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MineSubsidenceBackend")

app = FastAPI(
    title="SIH 26025 — Mine Subsidence Monitoring & Prediction API",
    description="Real-Time Underground Coal Mine Strata Subsidence, Time-to-Failure Prediction & Risk Early Warning System",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    WebSocket endpoint streaming live 1Hz telemetry and risk assessments directly to dashboards.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            latest = history_service.get_latest_reading()
            if latest:
                node_id = latest["node_id"]
                pred = prediction_service.predict_node_ttf(node_id)
                risk = risk_engine.assess_node_risk(node_id, pred)
                payload = {
                    "type": "TELEMETRY_UPDATE",
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_id": node_id,
                    "tilt_deg": latest.get("tilt_deg", 1.0),
                    "ttf_hours": pred.get("ttf_hours", 8.5),
                    "velocity_mm_h": pred.get("velocity_mm_h", 0.04),
                    "deformation_mm": pred.get("deformation_mm", 5.0),
                    "risk_state": risk.risk_state.value,
                    "risk_score": risk.risk_score,
                    "trend": risk.trend.value,
                    "confidence": risk.confidence
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
        "project": "SIH26025 - Mine Subsidence Monitoring & Prediction",
        "system": "Operational",
        "version": "1.0.0",
        "endpoints": {
            "telemetry": "/telemetry",
            "latest": "/latest",
            "prediction": "/prediction/{node_id}",
            "risk": "/risk",
            "risk_map": "/risk/map",
            "nodes": "/nodes",
            "alerts": "/alerts",
            "events": "/events",
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
        "mode": "OFFLINE_LOCAL_EDGE"
    }


@app.get("/system/status", response_model=SystemStatus, tags=["System"])
def get_system_status():
    latest_readings = history_service.get_all_latest_readings()
    from backend.storage.database import db
    active_alerts = db.get_active_alerts()
    crit_count = sum(1 for a in active_alerts if a.get("level") == "CRITICAL")

    return SystemStatus(
        system="Mine Subsidence Monitoring System",
        status="operational",
        mode="OFFLINE_LOCAL_EDGE",
        timestamp=datetime.utcnow(),
        active_nodes_count=9,
        online_nodes_count=len(latest_readings) if latest_readings else 9,
        critical_alerts_count=crit_count,
        overall_mine_risk=RiskState.NORMAL if crit_count == 0 else RiskState.CRITICAL,
        ml_model_loaded=prediction_service.is_ml_loaded
    )
