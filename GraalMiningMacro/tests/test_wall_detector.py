"""Unit tests for Wall Detector module."""

import numpy as np
import pytest
import cv2
from app.vision.wall_detector import WallDetector, WallDetection


def test_wall_detection_left():
    detector = WallDetector(confidence_threshold=0.30)
    # Create 200x200 frame with dark wall contour on left side
    frame = np.full((200, 200, 3), 150, dtype=np.uint8)
    frame[20:180, 10:50] = [10, 10, 10]  # Dark wall

    player_center = (120, 100)
    det = detector.detect(frame, player_center=player_center)

    assert isinstance(det, WallDetection)
    assert det.detected is True
    assert det.direction == "LEFT"
    assert det.distance_px > 0.0


def test_wall_detection_no_player():
    detector = WallDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    det = detector.detect(frame, player_center=None)
    assert det.detected is False
