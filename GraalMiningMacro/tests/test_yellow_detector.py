"""Unit tests for Yellow Glow Detector with temporal confirmation."""

import numpy as np
import pytest
from app.vision.yellow_detector import YellowGlowDetector, YellowGlowDetectionResult


def test_yellow_glow_temporal_confirmation():
    detector = YellowGlowDetector(min_area=30, required_frames=3)

    # Frame with bright yellow rock
    yellow_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    yellow_frame[20:60, 20:60] = [0, 255, 255]  # BGR Yellow

    # Frame 1: Accumulating (1/3)
    res1 = detector.detect(yellow_frame)
    assert res1.detected_raw is True
    assert res1.consecutive_frames == 1
    assert res1.is_confirmed is False

    # Frame 2: Accumulating (2/3)
    res2 = detector.detect(yellow_frame)
    assert res2.consecutive_frames == 2
    assert res2.is_confirmed is False

    # Frame 3: Confirmed (3/3)
    res3 = detector.detect(yellow_frame)
    assert res3.consecutive_frames == 3
    assert res3.is_confirmed is True


def test_yellow_glow_noise_rejection():
    detector = YellowGlowDetector(min_area=50, required_frames=3)

    # Dark/Blue frame without yellow
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dark_frame[:, :] = [255, 0, 0]  # Blue

    res = detector.detect(dark_frame)
    assert res.detected_raw is False
    assert res.is_confirmed is False
    assert res.consecutive_frames == 0
