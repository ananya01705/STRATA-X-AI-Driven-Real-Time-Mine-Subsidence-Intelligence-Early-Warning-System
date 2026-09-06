from fastapi import APIRouter, Query
from typing import List, Dict, Any

from backend.services.alert_service import alert_service
from backend.storage.database import db

router = APIRouter(tags=["Alerts & Events"])


@router.get("/alerts")
def get_alerts():
    """Returns all active, unresolved mine subsidence and sensor alerts."""
    return alert_service.get_active_alerts()


@router.get("/events")
def get_audit_events(limit: int = Query(50, description="Max event log items")):
    """Returns chronological audit log of system events, hazard warnings, and node transitions."""
    return db.get_recent_events(limit=limit)
