"""Unit tests for Status Infrastructure Detectors."""

import numpy as np
import pytest
from app.vision.status_detectors import StatusDetector, StatusDetectionResult, DrillState, BatteryState, MineLocationState


def test_status_detector_inside_mine():
    detector = StatusDetector()
    # Dark floor frame (inside mine)
    dark_frame = np.full((200, 200, 3), 30, dtype=np.uint8)

    res = detector.detect(dark_frame, player_detected=True)
    assert isinstance(res, StatusDetectionResult)
    assert res.drill_state == DrillState.EQUIPPED
    assert res.battery_state == BatteryState.BATTERY_OK
    assert res.mine_state == MineLocationState.INSIDE


def test_status_detector_surface():
    detector = StatusDetector()
    # Bright surface outdoor frame
    bright_frame = np.full((200, 200, 3), 180, dtype=np.uint8)

    res = detector.detect(bright_frame, player_detected=False)
    assert res.mine_state == MineLocationState.SURFACE
    assert res.drill_state == DrillState.UNKNOWN
