import os
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
from backend.services.history_service import history_service
from backend.storage.database import db
from backend.core.spatial_risk import spatial_risk_engine

try:
    import pytest  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    @pytest.fixture
    def client():
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def clean_history():
        """Cleans up in-memory history and test state before and after each test."""
        for nid in list(history_service._node_buffers.keys()):
            history_service.clear_node_history(nid)
        spatial_risk_engine._init_topology()
        yield
        for nid in list(history_service._node_buffers.keys()):
            history_service.clear_node_history(nid)
        spatial_risk_engine._init_topology()

except ImportError:
    pass
