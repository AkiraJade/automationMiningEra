"""Unit tests for Spider Threat Detector module."""

import numpy as np
import pytest
from app.vision.spider_detector import SpiderDetector, SpiderDetection


def test_spider_detector_empty_frame():
    detector = SpiderDetector()
    det = detector.detect(None)
    assert isinstance(det, SpiderDetection)
    assert det.detected is False
    assert det.distance_from_player == 0.0


def test_spider_detector_distance_calc():
    detector = SpiderDetector(confidence_threshold=0.50)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Red spider contour
    frame[10:25, 10:25] = [0, 0, 200]

    player_center = (100, 100)
    det = detector.detect(frame, player_center=player_center)
    if det.detected:
        assert det.distance_from_player > 0.0
