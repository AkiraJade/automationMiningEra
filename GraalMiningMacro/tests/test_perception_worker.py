"""Unit tests for PerceptionWorkerThread and multi-threaded pipeline responsiveness."""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication

from app.mining.perception_worker import PerceptionWorkerThread
from app.mining.mining_controller import MiningController
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.mining.mining_state import MiningState
from app.vision.message_detector import MessageDetector
from app.input.safety import safety


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def reset_safety_and_controller():
    safety.emergency_stopped = False
    yield
    safety.emergency_stopped = False


@pytest.fixture
def mock_controller():
    engine = MagicMock(spec=MiningPerceptionEngine)
    engine.process_frame.return_value = MiningPerceptionResult()
    controller = MiningController(perception_engine=engine)
    return controller


def test_perception_worker_initialization(mock_controller):
    worker = PerceptionWorkerThread(mining_controller=mock_controller, target_fps=10)
    assert worker.target_fps == 10
    metrics = worker.get_metrics()
    assert metrics["dropped_frames"] == 0
    assert metrics["health"] == "HEALTHY"


def test_latest_frame_buffer_overwrites_and_drops(mock_controller):
    worker = PerceptionWorkerThread(mining_controller=mock_controller, target_fps=10)
    
    frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 10
    frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 20
    frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 30

    # Enqueue multiple frames before worker consumes them
    worker.enqueue_frame(frame1)
    worker.enqueue_frame(frame2)
    worker.enqueue_frame(frame3)

    metrics = worker.get_metrics()
    # Frame 1 set available, Frame 2 and 3 overwrote -> 2 dropped frames
    assert metrics["dropped_frames"] == 2
    assert worker._latest_frame is not None
    assert np.array_equal(worker._latest_frame, frame3)


def test_perception_worker_execution_loop(qapp, mock_controller):
    worker = PerceptionWorkerThread(mining_controller=mock_controller, target_fps=20)
    received_results = []

    def on_complete(perception, proc_time_ms, perception_fps, health_status):
        received_results.append((perception, proc_time_ms, health_status))

    worker.perception_complete_signal.connect(on_complete)
    worker.start()

    frame = np.full((50, 50, 3), 100, dtype=np.uint8)
    worker.enqueue_frame(frame)

    time.sleep(0.2)
    qapp.processEvents()
    worker.stop()

    assert len(received_results) >= 1
    assert received_results[0][2] == "HEALTHY"


def test_perception_worker_exception_resilience(qapp):
    engine = MagicMock(spec=MiningPerceptionEngine)
    engine.process_frame.side_effect = RuntimeError("OpenCV crash")
    controller = MiningController(perception_engine=engine)

    worker = PerceptionWorkerThread(mining_controller=controller, target_fps=10)
    received_results = []

    def on_complete(perception, proc_time_ms, perception_fps, health_status):
        received_results.append((perception, health_status))

    worker.perception_complete_signal.connect(on_complete)
    worker.start()

    frame = np.full((50, 50, 3), 100, dtype=np.uint8)
    worker.enqueue_frame(frame)

    time.sleep(0.2)
    qapp.processEvents()
    worker.stop()

    assert len(received_results) >= 1
    assert controller.current_state == MiningState.PERCEPTION_ERROR
    assert received_results[0][1] == "ERROR"


def test_message_detector_cooldown_timestamp_based():
    detector = MessageDetector(default_cooldown_seconds=10.0)
    empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    res1 = detector.detect(empty_frame)
    assert res1.nothing_to_mine_detected is False

    # Simulate top banner detection by feeding a banner frame
    banner_frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # White horizontal banner in top 35% of frame
    banner_frame[20:40, 50:250] = 255
    
    t0 = time.time()
    res2 = detector.detect(banner_frame)
    assert res2.nothing_to_mine_detected is True
    assert res2.cooldown_until >= t0 + 9.9
    assert res2.cooldown_remaining > 0.0

    # Non-blocking timestamp check on empty frame
    res3 = detector.detect(empty_frame)
    assert res3.nothing_to_mine_detected is True
    assert res3.cooldown_remaining > 0.0
