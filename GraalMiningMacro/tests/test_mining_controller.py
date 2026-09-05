"""Unit tests for MiningController and state machine transition rules."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.mining.mining_controller import MiningController
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.mining.mining_state import MiningState
from app.vision.spider_detector import SpiderDetection
from app.vision.status_detectors import StatusDetectionResult, DrillState, BatteryState, MineLocationState
from app.vision.message_detector import MessageDetection
from app.vision.yellow_detector import YellowGlowDetectionResult
from app.vision.target_detector import TargetDetection
from app.input.safety import safety


@pytest.fixture
def mock_perception_engine():
    engine = MagicMock(spec=MiningPerceptionEngine)
    engine.process_frame.return_value = MiningPerceptionResult()
    return engine


@pytest.fixture
def controller(mock_perception_engine):
    # Reset safety state before each test
    safety.emergency_stopped = False
    return MiningController(perception_engine=mock_perception_engine)


def test_controller_initialization(controller):
    assert controller.current_state == MiningState.IDLE
    assert controller.automation_level == 1


def test_process_tick_normal_observation(controller, mock_perception_engine):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = controller.process_tick(frame)
    assert isinstance(res, MiningPerceptionResult)
    assert controller.current_state == MiningState.OBSERVATION


def test_process_tick_spider_detected(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        spider=SpiderDetection(detected=True, confidence=0.95)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.SPIDER_DETECTED


def test_process_tick_mine_collapsed(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        status=StatusDetectionResult(mine_state=MineLocationState.COLLAPSED)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.MINE_COLLAPSE_DETECTED


def test_process_tick_battery_empty(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        status=StatusDetectionResult(battery_state=BatteryState.BATTERY_EMPTY)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.BATTERY_EMPTY


def test_process_tick_drill_unequipped(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        status=StatusDetectionResult(drill_state=DrillState.UNEQUIPPED)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.CHECKING_DRILL


def test_process_tick_nothing_to_mine(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        message=MessageDetection(nothing_to_mine_detected=True, cooldown_remaining=5.0)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.NOTHING_TO_MINE


def test_process_tick_yellow_completed(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        yellow_glow=YellowGlowDetectionResult(is_confirmed=True)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller.process_tick(frame)

    assert controller.current_state == MiningState.TARGET_COMPLETED


def test_process_tick_target_detected_automation_levels(controller, mock_perception_engine):
    p_result = MiningPerceptionResult(
        target=TargetDetection(detected=True)
    )
    mock_perception_engine.process_frame.return_value = p_result

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Automation level 1 (Observation mode)
    controller.automation_level = 1
    controller.process_tick(frame)
    assert controller.current_state == MiningState.OBSERVATION

    # Automation level 2 (Mining active mode)
    controller.automation_level = 2
    controller.process_tick(frame)
    assert controller.current_state == MiningState.MINING


def test_process_tick_perception_exception_handling_and_debouncing(controller, mock_perception_engine):
    mock_perception_engine.process_frame.side_effect = RuntimeError("Simulated CV error")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Process first error tick
    res1 = controller.process_tick(frame)
    assert controller.current_state == MiningState.PERCEPTION_ERROR
    assert isinstance(res1, MiningPerceptionResult)

    # Process second error tick (debounced log, no crash)
    res2 = controller.process_tick(frame)
    assert controller.current_state == MiningState.PERCEPTION_ERROR
    assert isinstance(res2, MiningPerceptionResult)


def test_process_tick_emergency_stop(controller, mock_perception_engine):
    safety.emergency_stopped = True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    controller.process_tick(frame)

    assert controller.current_state == MiningState.EMERGENCY_STOP
