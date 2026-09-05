"""Unit tests for Mining Perception module."""

import numpy as np
import pytest
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.vision.color_detection import ColorDetector


def test_yellow_glow_color_detection():
    detector = ColorDetector()

    # Create a 200x200 image with a bright yellow circle
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bright yellow in BGR: B=0, G=255, R=255
    frame[50:100, 50:100] = [0, 255, 255]

    detections = detector.detect_yellow_glow_rocks(frame, min_area=30)
    assert len(detections) > 0
    assert detections[0].confidence > 0.0
    assert detections[0].bbox[2] > 0


def test_mining_perception_engine_process_empty_frame():
    engine = MiningPerceptionEngine()
    result = engine.process_frame(None)
    assert isinstance(result, MiningPerceptionResult)
    assert result.player.detected is False
