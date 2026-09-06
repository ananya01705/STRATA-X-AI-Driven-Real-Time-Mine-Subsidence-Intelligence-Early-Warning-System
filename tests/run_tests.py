"""
STRATA-X Standalone Test Runner
Executes all test functions across the test suite and reports test results.
"""

import sys
import traceback
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
from backend.services.history_service import history_service
from backend.core.spatial_risk import spatial_risk_engine

# Import test modules
import tests.test_conversions as tc
import tests.test_risk_engine as tr
import tests.test_spatial_risk as ts
import tests.test_api as ta
import tests.test_ml_fallback as tm


def run_all_tests():
    client = TestClient(app)
    
    test_modules = [
        ("Conversions", tc, [
            tc.test_compute_resultant_tilt,
            tc.test_transform_reading_to_ml_format_initial,
            tc.test_transform_reading_to_ml_format_with_previous,
            tc.test_transform_reading_zero_division_safeguard,
        ]),
        ("Risk Engine", tr, [
            tr.test_assess_node_risk_normal,
            tr.test_assess_node_risk_watch,
            tr.test_assess_node_risk_high,
            tr.test_assess_node_risk_critical,
            tr.test_assess_node_risk_sensor_fault_mitigation,
        ]),
        ("Spatial Risk Engine (Layer 2)", ts, [
            ts.test_spatial_topology_and_anchor,
            ts.test_neighbor_detection,
            ts.test_single_abnormal_node_anomaly_suppression,
            ts.test_connected_multi_node_risk_zone_detection,
            ts.test_multiple_independent_risk_clusters,
        ]),
        ("API Endpoints", ta, [
            lambda: ta.test_get_root(client),
            lambda: ta.test_get_health(client),
            lambda: ta.test_get_system_status(client),
            lambda: ta.test_post_telemetry_valid(client),
            lambda: ta.test_post_telemetry_invalid(client),
            lambda: ta.test_get_latest(client),
            lambda: ta.test_get_nodes(client),
            lambda: ta.test_get_node_details_valid(client),
            lambda: ta.test_get_node_details_invalid_404(client),
            lambda: ta.test_get_spatial_endpoints(client),
        ]),
        ("ML Fallback & Buffering", tm, [
            tm.test_prediction_insufficient_buffer,
            tm.test_prediction_full_buffer_inference,
        ]),
    ]

    total_passed = 0
    total_failed = 0

    print("================================================================")
    print("STRATA-X Automated Test Suite")
    print("================================================================")

    for module_name, _, test_funcs in test_modules:
        print(f"\n[MODULE] {module_name}")
        for func in test_funcs:
            # Clean history before each test
            for nid in list(history_service._node_buffers.keys()):
                history_service.clear_node_history(nid)
            spatial_risk_engine._init_topology()

            name = getattr(func, "__name__", str(func))
            try:
                func()
                print(f"  [PASS] {name}")
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}")
                traceback.print_exc()
                total_failed += 1

    print("\n================================================================")
    print(f"TEST SUMMARY: {total_passed} Passed, {total_failed} Failed (Total: {total_passed + total_failed})")
    print("================================================================")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
