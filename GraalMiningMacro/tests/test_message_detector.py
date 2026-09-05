"""Unit tests for Message Detector module."""

import numpy as np
import pytest
from app.vision.message_detector import MessageDetector, MessageDetection


def test_message_detector_no_message():
    detector = MessageDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    det = detector.detect(frame)
    assert isinstance(det, MessageDetection)
    assert det.nothing_to_mine_detected is False
    assert det.cooldown_remaining == 0.0


def test_message_detector_with_banner():
    detector = MessageDetector(default_cooldown_seconds=10.0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bright horizontal banner at top HUD
    frame[10:30, 20:180] = [255, 255, 255]

    det = detector.detect(frame)
    assert det.nothing_to_mine_detected is True
    assert det.cooldown_remaining > 0.0
