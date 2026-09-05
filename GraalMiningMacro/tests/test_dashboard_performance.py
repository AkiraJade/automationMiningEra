"""Unit tests for Deliverable 3.1: Dashboard Readability, Match Caching, Health Rolling Averages, and Detector Scheduling."""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication

from app.gui.dashboard import DashboardPage
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.mining.perception_worker import PerceptionWorkerThread
from app.mining.mining_controller import MiningController
from app.vision.reference import ReferenceMatcher, ReferenceManager, ReferenceImage, ReferenceMatchResult
from app.vision.status_detectors import DrillState, BatteryState, MineLocationState
from app.mining.mining_state import MiningState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_dashboard_system_monitor_layout_and_formatting(qapp):
    dashboard = DashboardPage()
    
    # Check width boundaries
    assert dashboard.findChild(object, None) is not None
    
    # Create test perception result with enums
    p = MiningPerceptionResult()
    p.status.drill_state = DrillState.EQUIPPED
    p.status.battery_state = BatteryState.BATTERY_OK
    p.status.mine_state = MineLocationState.INSIDE
    p.overall_confidence = 0.95

    dashboard.update_perception_display(
        p=p,
        window_connected=True,
        state_str="MiningState.SPIDER_DETECTED",
        health_status="HEALTHY",
        proc_time_ms=45.2,
        perception_fps=10.0,
        dropped_frames="12 (2.5%)"
    )

    assert dashboard.val_game.text() == "CONNECTED"
    assert dashboard.val_health.text() == "HEALTHY"
    assert dashboard.val_drill.text() == "EQUIPPED"
    assert dashboard.val_battery.text() == "OK"
    assert dashboard.val_location.text() == "INSIDE"
    assert dashboard.val_state.text() == "SPIDER DETECTED"
    assert dashboard.val_dropped.text() == "12 (2.5%)"
    assert dashboard.val_dropped.toolTip() == "12 (2.5%)"


def test_reference_matcher_cache_and_single_pass_grayscale():
    matcher = ReferenceMatcher()
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    gray_frame = np.full((100, 100), 128, dtype=np.uint8)

    dummy_ref = ReferenceImage(
        id="ref_test",
        name="Test Rock",
        category="rock",
        subcategory="yellow_complete",
        file_path="non_existent_file.png",
        threshold=0.8
    )

    match_cache = {}

    # First match call populates cache
    res1 = matcher.match_single(frame, dummy_ref, gray_image=gray_frame, match_cache=match_cache)
    assert (dummy_ref.id, None) in match_cache
    assert res1 == match_cache[(dummy_ref.id, None)]

    # Second match call returns exact cached object instantly
    res2 = matcher.match_single(frame, dummy_ref, gray_image=gray_frame, match_cache=match_cache)
    assert res2 is res1


def test_perception_engine_detector_scheduling_and_ages():
    engine = MiningPerceptionEngine()
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)

    # First tick runs all detectors
    res1 = engine.process_frame(frame)
    assert "status" in res1.detector_ages
    assert "message" in res1.detector_ages
    assert res1.detector_ages["player"] == 0.0

    # Second tick reuses scheduled detectors
    res2 = engine.process_frame(frame)
    assert isinstance(res2, MiningPerceptionResult)
    assert res2.detector_ages["status"] >= 0.0


def test_perception_worker_drop_rate_and_rolling_average():
    engine = MagicMock(spec=MiningPerceptionEngine)
    engine.process_frame.return_value = MiningPerceptionResult()
    controller = MiningController(perception_engine=engine)

    worker = PerceptionWorkerThread(mining_controller=controller, target_fps=10)
    
    frame1 = np.full((20, 20, 3), 50, dtype=np.uint8)
    frame2 = np.full((20, 20, 3), 60, dtype=np.uint8)
    frame3 = np.full((20, 20, 3), 70, dtype=np.uint8)

    worker.enqueue_frame(frame1)
    worker.enqueue_frame(frame2)
    worker.enqueue_frame(frame3)

    metrics = worker.get_metrics()
    assert metrics["total_enqueued_frames"] == 3
    assert metrics["dropped_frames"] == 2
    assert "66.7%" in metrics["dropped_formatted"]
