from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class StrataException(Exception):
    """Base exception for all STRATA-X domain errors."""
    def __init__(
        self,
        message: str,
        code: str = "STRATA_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class NodeNotFoundException(StrataException):
    def __init__(self, node_id: str):
        super().__init__(
            message=f"Node '{node_id}' is not registered in the mine spatial topology.",
            code="ERR_NODE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"node_id": node_id}
        )


class InsufficientTelemetryException(StrataException):
    def __init__(self, node_id: str, count: int, required: int = 30):
        super().__init__(
            message=f"Node '{node_id}' has {count}/{required} readings buffered. At least {required} readings are required for full LSTM inference.",
            code="ERR_INSUFFICIENT_TELEMETRY",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"node_id": node_id, "buffered_count": count, "required_count": required}
        )


class SensorFaultDetectedException(StrataException):
    def __init__(self, node_id: str, z_score: float):
        super().__init__(
            message=f"Isolated physical sensor fault detected on '{node_id}' (Z-Score: {z_score:.2f}).",
            code="ERR_SENSOR_FAULT",
            status_code=status.HTTP_409_CONFLICT,
            details={"node_id": node_id, "z_score": z_score}
        )


class PredictionUnavailableException(StrataException):
    def __init__(self, node_id: str, reason: str = "Inference engine offline or insufficient data"):
        super().__init__(
            message=f"Prediction unavailable for node '{node_id}': {reason}",
            code="ERR_PREDICTION_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"node_id": node_id, "reason": reason}
        )


class SpatialDataUnavailableException(StrataException):
    def __init__(self, reason: str = "Spatial engine topology uninitialized"):
        super().__init__(
            message=f"Spatial risk analysis unavailable: {reason}",
            code="ERR_SPATIAL_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"reason": reason}
        )


async def strata_exception_handler(request: Request, exc: StrataException) -> JSONResponse:
    logger.error(f"StrataException [{exc.code}] on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "code": exc.code,
            "detail": exc.message,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
            "details": exc.details
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled Exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred within the STRATA-X gateway.",
            "code": "ERR_INTERNAL_SERVER_ERROR",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )
