"""Unit tests for Player Detector module."""

import numpy as np
import pytest
from app.vision.player_detector import PlayerDetector, PlayerDetection


def test_player_detection_centered():
    detector = PlayerDetector(confidence_threshold=0.65)
    # Create 200x200 test frame
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    det = detector.detect(frame)
    assert isinstance(det, PlayerDetection)
    assert det.detected is True
    assert det.center == (100, 100)
    assert det.confidence >= 0.65
    assert det.movement_delta == 0.0


def test_player_movement_delta():
    detector = PlayerDetector(confidence_threshold=0.50)
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2 = np.zeros((300, 300, 3), dtype=np.uint8)

    det1 = detector.detect(frame1)
    assert det1.center == (100, 100)

    det2 = detector.detect(frame2)
    assert det2.center == (150, 150)
    assert det2.movement_delta > 0.0


def test_player_detection_empty_frame():
    detector = PlayerDetector()
    det = detector.detect(None)
    assert det.detected is False
