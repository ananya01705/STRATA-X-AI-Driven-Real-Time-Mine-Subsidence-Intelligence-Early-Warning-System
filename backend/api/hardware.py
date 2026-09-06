from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.services.serial_service import serial_service

router = APIRouter(prefix="/hardware", tags=["Hardware & Serial Ingestion"])


class SerialConnectRequest(BaseModel):
    port: str = Field(default="COM3", description="Serial COM port (e.g. COM3 or /dev/ttyUSB0)")
    baudrate: int = Field(default=115200, description="Baud rate (default: 115200)")


@router.get("/ports", response_model=List[Dict[str, Any]])
def get_available_ports():
    """List all detected USB Serial COM ports on the host system."""
    return serial_service.list_available_ports()


@router.get("/serial/status")
def get_serial_status():
    """Retrieve status, packet metrics, and health of the hardware serial listener."""
    return serial_service.get_status()


@router.post("/serial/start")
def start_serial_listener(req: SerialConnectRequest):
    """Start listening on physical ESP32 / LoRa gateway USB port."""
    success = serial_service.start(port=req.port, baudrate=req.baudrate)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not start serial service (pyserial may not be installed or port is inaccessible)."
        )
    return {"message": f"Serial listener active on {req.port} at {req.baudrate} baud.", "status": "RUNNING"}


@router.post("/serial/stop")
def stop_serial_listener():
    """Stop the hardware serial background listener."""
    serial_service.stop()
    return {"message": "Serial listener stopped.", "status": "STOPPED"}
